# coding: utf-8
import os
import time
import numpy as np

import tensorflow
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from idlelib.calltip import get_entity
from dataset import random_embedding, read_corpus, read_dictionary, tag2label
from model import BiLSTM_CRF


if __name__ == '__main__':
    tf.flags.DEFINE_string('train_data', 'data', 'train data source')
    tf.flags.DEFINE_string('test_data', 'data', 'test data source')
    tf.flags.DEFINE_integer('batch_size', 64, 'sample of each minibatch')

    tf.flags.DEFINE_integer('epoch', 15, 'epoch of traing')
    tf.flags.DEFINE_integer('hidden_dim', 300, 'dim of hidden state')
    tf.flags.DEFINE_string('optimizer', 'Adam', 'Adam/Adadelta/Adagrad/RMSProp/Momentum/SG')

    tf.flags.DEFINE_boolean('CRF', False, 'use CRF at the top layer. if False, use Softmax')
    tf.flags.DEFINE_float('lr', 0.001, 'learing rate')
    tf.flags.DEFINE_float('clip', 5.0, 'gradient clipping')
    tf.flags.DEFINE_float('dropout', 0.5, 'dropout keep_prob')

    tf.flags.DEFINE_boolean('update_embeddings', True, 'update embeddings during traings')
    tf.flags.DEFINE_string('pretrain_embedding', 'random', 'use pretrained char embedding or init it randomly')
    tf.flags.DEFINE_integer('embedding_dim', 300, 'random init char embedding_dim')

    tf.flags.DEFINE_boolean('shuffle', True, 'shuffle training data before each epoch')
    tf.flags.DEFINE_string('mode', 'train', 'train/test/demo')
    tf.flags.DEFINE_string('ckpt_path', 'ckpts', 'save model path')
    tf.flags.DEFINE_string('log_path', 'log', 'save summary path')
    args = tf.flags.FLAGS


    word2id = read_dictionary(os.path.join(args.train_data, 'word2id.pkl'))
    if args.pretrain_embedding == 'random':
        embeddings = random_embedding(word2id, args.embedding_dim)
    else:
        embedding_path = 'pretrain_embedding.npy'
        embeddings = np.array(np.load(embedding_path), dtype='float32')


    if args.mode != 'demo':
        train_path = os.path.join(args.train_data, 'train_data1')
        test_path = os.path.join(args.test_data, 'test_data1')
        print(train_path, test_path)
        train_data = read_corpus(train_path)
        test_data = read_corpus(test_path)
        test_size = len(train_data)
        print(test_size)

    ## paths setting
    model_path = args.ckpt_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, 'model')

    summary_path = args.log_path
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    ## training model
    if args.mode == 'train':
        print('==========lr====', args.lr)
        model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                           dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip, tag2label=tag2label,
                           vocab=word2id, shuffle=args.shuffle, model_path=ckpt_prefix, summary_path=summary_path, CRF=args.CRF,
                           update_embedding=args.update_embeddings)
        model.build_graph()
        print('train data len=', len(train_data))
        model.train(train_data, test_data)

    elif args.mode == 'test':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                           dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip, tag2label=tag2label,
                           vocab=word2id, shuffle=args.shuffle, model_path=ckpt_file, summary_path=summary_path, CRF=args.CRF,
                           update_embedding=args.update_embedding)
        model.build_graph()
        print('test data: {}'.format(test_size))
        model.test(test_data)

    elif args.mode == 'demo':
        ckpt_file = tf.train.latest_checkpoint(model_path)
        print(ckpt_file)
        model = BiLSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                           dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip, tag2label=tag2label,
                           vocab=word2id, shuffle=args.shuffle, model_path=ckpt_file, summary_path=summary_path, CRF=args.CRF,
                           update_embedding=args.update_embeddings)
        model.build_graph()
        saver = tf.train.Saver()
        with tf.Session as sess:
            saver.restore(sess, ckpt_file)
            while True:
                print('Please input your sentence:')
                demo_sent = input()
                if demo_sent == '' or demo_sent.isspace():
                    print('See you next time!')
                    break
                else:
                    demo_sent = list(demo_sent.strip())
                    demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                    tag = model.demo_one(sess, demo_data)
                    PER, LOC, ORG = get_entity(tag, demo_sent)
                    print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
