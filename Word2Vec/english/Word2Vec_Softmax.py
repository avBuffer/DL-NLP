# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data


def make_data(skip_grams):
    input_data = []
    output_data = []
    for a, b in skip_grams:
        input_data.append(np.eye(vocab_size)[a])
        output_data.append(b)
    return input_data, output_data


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(vocab_size, m).type(dtype))
        self.V = nn.Parameter(torch.randn(m, vocab_size).type(dtype))

    def forward(self, X):
        # X : [batch_size, vocab_size]
        hidden = torch.mm(X, self.W) # [batch_size, m]
        output = torch.mm(hidden, self.V) # [batch_size, vocab_size]
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.FloatTensor

    is_only_text = True
    if is_only_text:
        sentences = ["jack like dog", "jack like cat", "jack like animal", "dog cat animal", "banana apple cat dog like",
                     "dog fish milk like", "dog cat animal like", "jack like apple", "apple like", "jack like banana",
                     "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
        sentence_list = " ".join(sentences).split() # ['jack', 'like', 'dog']

    else:
        with open('data/text8.train.txt') as f:
            sentences = f.read() # 得到文本内容
        all_sentence_list = sentences.lower().split() #　分割成单词列表
        split_index = int(len(all_sentence_list) * 0.01)
        sentence_list = all_sentence_list[:split_index]
        print('sentences.len=', len(sentences), 'all_sentence_list.len=', len(all_sentence_list), 'sentence_list.len=', len(sentence_list))

    vocab = list(set(sentence_list))
    word2idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    print('vocab_size=', vocab_size)

    # model parameters
    C = 2 # window size
    batch_size = 8
    m = 2 # word embedding dim
    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    skip_grams = []
    for idx in range(C, len(sentence_list) - C):
        center = word2idx[sentence_list[idx]]
        context_idx = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        context = [word2idx[sentence_list[i]] for i in context_idx]
        for w in context:
            skip_grams.append([center, w])
    print('skip_grams.len=', len(skip_grams))

    input_data, output_data = make_data(skip_grams)
    print('make_data input_data.len=', len(input_data), 'output_data.len=', len(output_data))
    input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
    print('torch input_data.len=', len(input_data), 'output_data.len=', len(output_data))

    dataset = Data.TensorDataset(input_data, output_data)
    loader = Data.DataLoader(dataset, batch_size, True)

    model = Word2Vec().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = optimizer.Adam(model.parameters(), lr=1e-3)

    for epoch in range(max_iter):
        for idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "embedding_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optim.zero_grad()
            loss.backward()
            optim.step()

    for idx, label in enumerate(vocab):
        W, WT = model.parameters()
        print('idx=', idx, 'label=', label, 'W=', W.shape, 'WT=', WT.shape)
        x, y = float(W[idx][0]), float(W[idx][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    #plt.show()
    plt.savefig(os.path.join(ckpt_path, 'Word2Vec.png'))
