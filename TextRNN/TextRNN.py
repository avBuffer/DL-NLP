# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def make_data(sentences):
    input_batch = []
    target_batch = []
    for sen in sentences:
        word = sen.split()
        input = [word2idx[n] for n in word[:-1]]
        target = word2idx[word[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return input_batch, target_batch


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        # input_size 指的是每个单词用多少维的向量去编码
        # hidden_size 指的是输出维度是多少
        # fully connected layer
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, hidden, X):
        # X: [batch_size, n_step, n_class]
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        out, hidden = self.rnn(X, hidden)
        # out : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        out = out[-1] # [batch_size, num_directions(=1) * n_hidden] ⭐
        model = self.fc(out)
        return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.FloatTensor

    sentences = ["i like dog", "i love coffee", "i hate milk"]
    word_list = " ".join(sentences).split()
    vocab = list(set(word_list))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    n_class = len(vocab)

    # TextRNN Parameter
    batch_size = 2
    n_step = 2 # number of cells(= number of Step)
    n_hidden = 5 # number of hidden units in one cell

    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    input_batch, target_batch = make_data(sentences)
    print('make_data input_batch.len=', len(input_batch), 'target_batch.len=', len(target_batch))
    input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)
    print('torch input_batch.len=', len(input_batch), 'target_batch.len=', len(target_batch))
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)

    model = TextRNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(max_iter):
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            # hidden : [num_layers * num_directions, batch, hidden_size]
            hidden = torch.zeros(1, x.shape[0], n_hidden).to(device) # h0
            # x : [batch_size, n_step, n_class]
            pred = model(hidden, x)

            # pred : [batch_size, n_class], y : [batch_size] (LongTensor, not one-hot)
            loss = criterion(pred, y)
            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "textnn_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Predict
    input = [sen.split()[:2] for sen in sentences]
    hidden = torch.zeros(1, len(input), n_hidden).to(device)
    predict = model(hidden, input_batch.to(device)).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [idx2word[n.item()] for n in predict.squeeze()])
