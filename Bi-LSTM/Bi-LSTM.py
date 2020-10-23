# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def make_data(sentence):
    input_batch = []
    target_batch = []
    words = sentence.split() # ['Github', 'Actions', 'makes', ...]

    for i in range(max_len - 1): # i = 2
        input = [word2idx[n] for n in words[:(i + 1)]] # input = [18 7 3]
        input = input + [0] * (max_len - len(input)) # input = [18 7 3 0 'it', ..., 0]
        target = word2idx[words[i + 1]] # target = [0]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        # fc
        self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]
        hidden_state = torch.randn(1 * 2, batch_size, n_hidden).to(device)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size, n_hidden).to(device)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state)) # [max_len, batch_size, n_hidden * 2]
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.FloatTensor

    sentence = ('GitHub Actions makes it easy to automate all your software workflows '
                'from continuous integration and delivery to issue triage and more')

    word2idx = {w: i for i, w in enumerate(list(set(sentence.split())))}
    idx2word = {i: w for i, w in enumerate(list(set(sentence.split())))}
    n_class = len(word2idx) # classification problem

    batch_size = 16
    max_len = len(sentence.split())
    n_hidden = 5
    
    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # input_batch: [max_len - 1, max_len, n_class]
    input_batch, target_batch = make_data(sentence)
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)

    model = BiLSTM().to(device)
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
                model_file = os.path.join(ckpt_path, "bi-lstm_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Pred
    predict = model(input_batch.to(device)).data.max(1, keepdim=True)[1]
    results = [idx2word[n.item()] for n in predict.squeeze()]
    print('sentence=', sentence, '\n', 'results=', results)
    for idx, result in enumerate(results):
        print('idx=', idx, 'result=', result)
