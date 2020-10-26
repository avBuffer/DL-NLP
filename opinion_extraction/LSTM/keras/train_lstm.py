#coding:utf-8
import os
import numpy as np
import gensim
import string
from gensim.models.word2vec import LineSentence
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import Model

all_words = {}
wtf_words = {} 


class Config(object):
    data_path = '../../data'
    embedding_path = '../../embedding'

    MAX_SEQUENCE_LENGTH = 15
    EMBEDDING_DIM = 200
    HIDDEN_SIZE = 80
    max_iter = 5

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
                line_vecs.append(np.zeros(config.EMBEDDING_DIM))
                if word not in wtf_words:
                    wtf_words[word] = ''

        if len(line_vecs) > config.MAX_SEQUENCE_LENGTH:
            line_vecs = line_vecs[:config.MAX_SEQUENCE_LENGTH]
        else:
            line_vecs = [np.zeros(config.EMBEDDING_DIM)] * (config.MAX_SEQUENCE_LENGTH - len(line_vecs)) + line_vecs
        vecs.append(line_vecs)
    return np.array(vecs), lines


def load_label_to_vecs(path):
    lines = open(path).read().split('\n')
    vecs = []
    for line in lines:
        labels = []
        for label in line.split(' '):
            if label == '0':
                labels.append([1, 0, 0])
            elif label == '1':
                labels.append([0, 1, 0])
            else:
                labels.append([0, 0, 1])

        if len(labels) > config.MAX_SEQUENCE_LENGTH:
            labels = labels[:config.MAX_SEQUENCE_LENGTH]
        else:
            labels = [[0, 0, 0]] * (config.MAX_SEQUENCE_LENGTH - len(labels)) + labels
        vecs.append(labels)
    return np.array(vecs)


def train_d2v_model(infile, embedding_file):
    model = gensim.models.Word2Vec(LineSentence(infile), size=200, window=5, min_count=5)
    model.save(embedding_file)


if __name__ == '__main__':
    config = Config()

    print('(1)load data...')
    embedding_file = os.path.join(config.embedding_path, 'yelp.vector.bin')
    if not os.path.exists(embedding_file):
        train_d2v_model(os.path.join(config.data_path, 'train_docs.txt'), embedding_file)
        print('embedding_file not exist and then trained')

    x_train, x_docs = load_doc_to_vecs(os.path.join(config.data_path, 'train_docs.txt'), embedding_file)
    y_train_a = load_label_to_vecs(os.path.join(config.data_path, 'train_labels_a.txt'))
    y_train_p = load_label_to_vecs(os.path.join(config.data_path, 'train_labels_p.txt'))
    print('there are ' + str(len(all_words)) + ' words totally')
    print('there are ' + str(len(wtf_words)) + ' words not be embeded')
    print('train docs:' + str(x_train.shape))
    print('train labels of aspect:' + str(y_train_a.shape))
    print('train labels of opinion:' + str(y_train_p.shape))


    print('(2)build model...')
    main_input = Input(shape=(config.MAX_SEQUENCE_LENGTH, config.EMBEDDING_DIM), dtype='float32')
    lstm1 = LSTM(config.HIDDEN_SIZE, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, name='lstm1')(main_input)
    out1 = Dense(3, activation='softmax', name='out1')(lstm1)
    model = Model(inputs=main_input, outputs=out1)
    model.summary()


    print('(3)run model...')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.fit(x_train, y_train_a, epochs=config.max_iter, batch_size=32)
    model.save(os.path.join(config.ckpt_path, 'lstm_model.h5'))
