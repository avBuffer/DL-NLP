# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def make_data(seq_data):
    input_batch, target_batch = [], []
    for seq in seq_data:
        input = [word2idx[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word2idx[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        # fc
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, X):
        # X: [batch_size, n_step, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        hidden_state = torch.zeros(1, batch_size, n_hidden).to(device)   # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, batch_size, n_hidden).to(device)    # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.fc(outputs)
        return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.FloatTensor

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz'] # ['a', 'b', 'c',...]
    word2idx = {n: i for i, n in enumerate(char_arr)}
    idx2word = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word2idx) # number of class(=number of vocab)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    # TextLSTM Parameters
    batch_size = 3
    n_step = len(seq_data[0]) - 1 # (=3)
    n_hidden = 128

    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    input_batch, target_batch = make_data(seq_data)
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)

    model = TextLSTM().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(max_iter):
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "textlstm_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    inputs = [sen[:3] for sen in seq_data]
    predict = model(input_batch.to(device)).data.max(1, keepdim=True)[1]
    print(inputs, '->', [idx2word[n.item()] for n in predict.squeeze()])
