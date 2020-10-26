import os
import gensim
import numpy as np
import time
import string

import tensorflow
print('tensorflow.version=', tensorflow.__version__)
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from gensim.models.word2vec import LineSentence
from cmlapp import CMLA

all_words = {}
wtf_words = {}   


class Config(object):
    data_path = '../../data'
    embedding_path = '../../embedding'

    embedding_dim = 200
    attention_slice = 15
    gru_hidden_size = 30
    batch_size = 1
    num_layer = 2
    learning_rate = 0.0007
    drop_rate = 0.5
    max_grad_norm = 5

    max_iter = 5
    statistic_step = int(max_iter * 0.1) if int(max_iter * 0.1) > 0 else 2
    test_batch_size = 128

    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    
def load_doc_to_vecs(path, embedding_file):
    model = gensim.models.Word2Vec.load(embedding_file)
    lines = open(path).read().split('\n')
    vecs = []

    for line in lines:
        line = line.translate(string.punctuation)
        words = line.split(' ')
        line_vecs = []
        for word in words:
            if word not in all_words:
                all_words[word] = ''
            if word in model:
                line_vecs.append(model[word])
            else:
                line_vecs.append(np.zeros(config.embedding_dim))
                if word not in wtf_words:
                    wtf_words[word] = ''
        vecs.append(line_vecs)
    return vecs, lines


def load_label_to_vecs(path):
    lines = open(path).read().split('\n')
    vecs = []
    for line in lines:
        labels = line.split(' ')
        vecs.append(labels)
    return vecs


def train_d2v_model(infile, embedding_file):
    model = gensim.models.Word2Vec(LineSentence(infile), size=200, window=5, min_count=5)
    model.save(embedding_file)


if __name__ == '__main__':
    config = Config()
    
    print('(1) load data and trans to vecs...')
    embedding_file = os.path.join(config.embedding_path, 'yelp.vector.bin')
    if not os.path.exists(embedding_file):
        train_d2v_model(os.path.join(config.data_path, 'train_docs.txt'), embedding_file)
        print('embedding_file not exist and then trained')
    
    x_train, x_docs = load_doc_to_vecs(os.path.join(config.data_path, 'train_docs.txt'), embedding_file)
    y_train_a = load_label_to_vecs(os.path.join(config.data_path, 'train_labels_a.txt'))
    y_train_p = load_label_to_vecs(os.path.join(config.data_path, 'train_labels_p.txt'))
    x_train = np.array(x_train)

    x_test, x_docs_test = load_doc_to_vecs(os.path.join(config.data_path, 'test_docs.txt'), embedding_file)
    y_test_a = load_label_to_vecs(os.path.join(config.data_path, 'test_labels_a.txt'))
    y_test_p = load_label_to_vecs(os.path.join(config.data_path, 'test_labels_p.txt'))
    x_test = np.array(x_test)

    print('there are ' + str(len(all_words))+' words totally')
    print('there are ' + str(len(wtf_words))+' words not be embeded')
    print('train docs:' + str(len(x_train)), 'train labels of aspect:' + str(len(y_train_a)), 'train labels of opinion:' + str(len(y_train_p)))
    print('test docs:' + str(len(x_test)), 'test labels of aspect:' + str(len(y_test_a)), 'test labels of opinion:' + str(len(y_test_p)))
        

    print('(2) build model...')
    model = CMLA(config=config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    

    print('(3) train model...')
    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        #tf.summary.FileWriter('graph', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        start = time.time()
        new_state = sess.run(model.gru_init_state)
        total_loss = 0
        less_loss = 10000.0
        for epoch in range(config.max_iter):
            for i in range(len(x_train)):
                feed_dict = {model.x: x_train[i], model.y1: y_train_a[i], model.y2: y_train_p[i]}
                for ii, dd in zip(model.gru_init_state, new_state):
                    feed_dict[ii] = dd

                loss, new_state, _ = sess.run([model.loss, model.gru_final_state, model.optimizer], feed_dict=feed_dict)
                total_loss += loss
                end = time.time()

                if loss < less_loss:
                    model_file = os.path.join(config.ckpt_path, "cmla_tf_{}-{}-{}.th".format(epoch, i, loss))
                    saver.save(sess, model_file, global_step=epoch)
                    print('epoch:' + str(epoch), 'steps:' + str(i), 'model_file=', model_file)
                    less_loss = loss

                if i % config.statistic_step == 0:
                    ave_loss = total_loss
                    if i > 0:
                        ave_loss = total_loss / config.statistic_step
                    print('epoch:' + str(epoch) + ' / ' + str(config.max_iter), 'steps: ' + str(i),
                          'cost_time:' + str(end - start), 'loss:' + str(ave_loss))

                    total_loss = 0
                    correct_a_num = 0
                    correct_p_num = 0
                    for index in range(config.test_batch_size):
                        feed_dict[model.x] = x_train[index]
                        feed_dict[model.y1] = y_train_a[index]
                        feed_dict[model.y2] = y_train_p[index]
                        correct_a, correct_p = sess.run([model.correct_a, model.correct_p], feed_dict=feed_dict)
                        if correct_a:
                            correct_a_num += 1
                        if correct_p:
                            correct_p_num += 1
                    score1 = float(correct_a_num) * 100 / config.test_batch_size
                    score2 = float(correct_p_num) * 100 / config.test_batch_size
                    print('Train epoch:' + str(epoch), 'steps:' + str(i), 'precision: ' + str(score1) + ' ' + str(score2))

                    correct_a_num = 0
                    correct_p_num = 0
                    test_batch_size = len(x_test)
                    for index in range(test_batch_size):
                        feed_dict[model.x] = x_test[index]
                        feed_dict[model.y1] = y_test_a[index]
                        feed_dict[model.y2] = y_test_p[index]
                        correct_a, correct_p = sess.run([model.correct_a, model.correct_p], feed_dict=feed_dict)
                        if correct_a:
                            correct_a_num += 1
                        if correct_p:
                            correct_p_num += 1
                    score1 = float(correct_a_num) * 100 / config.test_batch_size
                    score2 = float(correct_p_num) * 100 / config.test_batch_size
                    print('Test epoch:' + str(epoch), 'steps:' + str(i), 'precision: ' + str(score1) + ' ' + str(score2))
