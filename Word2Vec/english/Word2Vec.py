import os
import numpy as np
import random
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word'''
        super(WordEmbeddingDataset, self).__init__() # #通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text] # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded) # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word'''
        center_words = self.text_encoded[idx] # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices] # tensor(list)
        
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        
        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(pos_indices) & set(neg_words)) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        return center_words, pos_words, neg_words


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            return: loss, [batch_size]'''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]
        
        log_pos = F.logsigmoid(pos_dot).sum(1) # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg
        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()


def find_nearest(word):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    C = 3 # context window
    K = 15 # number of negative samples
    MAX_VOCAB_SIZE = 10000
    EMBEDDING_SIZE = 100
    batch_size = 32
    lr = 0.2

    max_iter = 5
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    with open('data/text8.train.txt') as f:
        text = f.read() # 得到文本内容

    text = text.lower().split() #　分割成单词列表
    vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1)) # 得到单词字典表，key是单词，value是次数
    vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values())) # 把不常用的单词都编码为"<UNK>"

    word2idx = {word : i for i, word in enumerate(vocab_dict.keys())}
    idx2word = {i : word for i, word in enumerate(vocab_dict.keys())}

    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3./4.)

    dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
    dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)

    model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(max_iter):
        for idx, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "word2vec_{}-{}-{}.th".format(epoch, EMBEDDING_SIZE, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

    embedding_weights = model.input_embedding()
    for word in ["two", "america", "computer"]:
        print(word, find_nearest(word))
