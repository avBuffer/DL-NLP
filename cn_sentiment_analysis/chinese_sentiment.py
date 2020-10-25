#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import re
import jieba
import bz2
import warnings
warnings.filterwarnings('ignore')

# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# 反向tokenize, 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text


def predict_sentiment(text, conf_thresh):
    text = re.sub('[\s+\.\!\/_,$%^*(+\'\']+|[+——！，。？、~@#￥%……&*（）]+', '', text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
            if cut_list[i] >= num_words:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0

    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens, padding='pre', truncating='pre')
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= conf_thresh:
        print(text + '是一例正面评价', 'output=%.2f' % coef)
    else:
        print(text + '是一例负面评价', 'output=%.2f' % coef)
    return coef


if __name__ == '__main__':
    # 使用gensim加载预训练中文分词embedding
    embeddings_file = 'embedding/sgns.zhihu.bigram'
    cn_model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False, unicode_errors='ignore')
    embedding_dim = cn_model['自然语言处理'].shape[0]
    print('词向量的长度为{}'.format(embedding_dim), '自然语言处理=', cn_model['自然语言处理'].shape)

    train_texts_orig = []
    train_target = []

    pos_file = 'data/positive_samples.txt'
    with open(pos_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            dic = eval(line)
            train_texts_orig.append(dic['text'])
            train_target.append(dic['label'])

    neg_file = 'data/negative_samples.txt'
    with open(neg_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            dic = eval(line)
            train_texts_orig.append(dic['text'])
            train_target.append(dic['label'])

    print('train_texts_orig.len=', len(train_texts_orig), 'train_target.len=', len(train_target))


    # 进行分词和tokenize
    train_tokens = []
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub('[\s+\.\!\/_,$%^*(+\'\']+|[+——！，。？、~@#￥%……&*（）]+', '', text)
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器, 把生成器转换为list
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        train_tokens.append(cut_list)


    # 索引长度标准化
    # 因为每段评语的长度是不一样的，我们如果单纯取最长的一个评语，并把其他评填充成同样的长度，这样十分浪费计算资源，所以我们取一个折衷的长度。
    # 获得所有tokens的长度
    num_tokens = [len(tokens) for tokens in train_tokens]
    num_tokens = np.array(num_tokens)
    # 平均tokens的长度, 最长的评价tokens的长度
    print('num_tokens.mean=', np.mean(num_tokens), 'num_tokens.max=', np.max(num_tokens))

    # 取tokens平均值并加上两个tokens的标准差，假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    print('max_tokens=', max_tokens)

    # 取tokens的长度为236时，大约95%的样本被涵盖, 我们对长度不足的进行padding，超长的进行修剪
    np.sum(num_tokens < max_tokens) / len(num_tokens)

    # 准备Embedding Matrix
    # 为模型准备embedding matrix（词向量矩阵），一个维度为$(numwords, embeddingdim)$的矩阵，
    # num words代表我们使用的词汇的数量，emdedding dimension在我们现在使用的预训练词向量模型中是300，每一个词汇都用一个长度为300的向量表示。
    # 注意我们只选择使用前50k个使用频率最高的词，在这个预训练词向量模型中，一共有260万词汇量，如果全部使用在分类问题上会很浪费计算资源，
    num_words = 50000
    # 初始化embedding_matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵, 维度为 50000 * 300
    for i in range(num_words):
        embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    print('embedding_matrix.shape=', embedding_matrix.shape)

    # **padding（填充）和truncating（修剪）**
    # 我们把文本转换为tokens（索引）之后，每一串索引的长度并不相等，所以为了方便模型的训练我们需要把索引的长度标准化，上面我们选择了236这个可以涵盖95%训练样本的长度，
    # 接下来我们进行padding和truncating，我们一般采用'pre'的方法，这会在文本索引的前面填充0，因为根据一些研究资料中的实践，如果在文本索引后面填充0的话，会对模型造成一些不良影响。
    # 进行padding和truncating，输入的train_tokens是一个list，返回的train_pad是一个numpy array
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')

    # 超出五万个词向量的词用0代替
    train_pad[train_pad >= num_words] = 0
    # 准备target向量，前2000样本为1，后2000为0
    train_target = np.array(train_target)
    # 90%的样本用来训练，剩余10%用来测试
    X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=0.1, random_state=12)

    # 构建模型
    # GRU：GRU最后一层激活函数的输出都在0.5左右，说明模型的判断不是很明确、信心比较低，我们期望对于负面样本输出接近0，正面样本接近1而不是都徘徊于0.5之间。
    # BiLSTM：LSTM的表现略好于GRU，这可能是因为BiLSTM对于比较长的句子结构有更好的记忆。
    is_lstm = True
    model = Sequential()

    if is_lstm:
        # Embedding之后第一层我们用BiLSTM返回sequences，然后第二层16个单元的LSTM不返回sequences，只返回最终结果，最后是一个全链接层，用sigmoid激活函数输出结果。
        # 现在我们用keras搭建LSTM模型，模型的第一层是Embedding层，只有当我们把tokens索引转换为词向量矩阵之后，才可以用神经网络对文本进行处理。
        # 在Embedding层我们输入的矩阵为：$$(batchsize, maxtokens)$$, 输出矩阵为： $$(batchsize, maxtokens, embeddingdim)$$
        model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_tokens, trainable=False))
        model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        model.add(LSTM(units=16, return_sequences=False))
    else:
        # GRU的代码
        model.add(GRU(units=32, return_sequences=True))
        model.add(GRU(units=16, return_sequences=True))
        model.add(GRU(units=4, return_sequences=False))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    if not os.path.exists('ckpts'):
        os.makedirs('ckpts')

    path_checkpoint = 'ckpts/sentiment'
    checkpoint = ModelCheckpoint(filepath=path_checkpoint + '_{epoch:05d}_{val_loss:.8f}', monitor='val_loss',
                                 verbose=1, save_weights_only=True, save_best_only=False)
    try:
        model.load_weights(path_checkpoint)
    except Exception as e:
        print('exeption=', e)

    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-8, patience=0, verbose=1)
    callbacks = [checkpoint, lr_reduction]

    model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=128, callbacks=callbacks)
    result = model.evaluate(X_test, y_test)
    print('Accuracy:{0:.2%}'.format(result[1]), 'result=', result)
