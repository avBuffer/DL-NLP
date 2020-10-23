# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data


def make_data(sentences):
    input_data = []
    target_data = []
    for sen in sentences:
        sen = sen.split() # ['i', 'like', 'cat']
        input_tmp = [word2idx[w] for w in sen[:-1]]
        target_tmp = word2idx[sen[-1]]
        input_data.append(input_tmp)
        target_data.append(target_tmp)
    return input_data, target_data


# parameters
m = 2
n_step = 2
n_hidden = 10

class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(V, m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.b = nn.Parameter(torch.randn(V).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, V).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, V).type(dtype))

    def forward(self, X):
        '''X : [batch_size, n_step]'''
        X = self.C(X) # [batch_size, n_step, m]
        X = X.view(-1, n_step * m) # [batch_szie, n_step * m]
        hidden_out = torch.tanh(self.d + torch.mm(X, self.H)) # [batch_size, n_hidden]
        output = self.b + torch.mm(X, self.W) + torch.mm(hidden_out, self.U)
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.FloatTensor

    sentences = ['i like cat', 'i love coffee', 'i hate milk']
    sentences_list = " ".join(sentences).split() # ['i', 'like', 'cat', 'i', 'love'. 'coffee',...]
    vocab = list(set(sentences_list))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    
    V = len(vocab)
    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    input_data, target_data = make_data(sentences)
    input_data, target_data = torch.LongTensor(input_data), torch.LongTensor(target_data)
    dataset = Data.TensorDataset(input_data, target_data)
    loader = Data.DataLoader(dataset, 2, True)
    
    model = NNLM().to(device)
    optim = optimizer.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(max_iter):
        for idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "nnlm_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optim.zero_grad()
            loss.backward()
            optim.step()

    # Pred
    pred = model(input_data.to(device)).max(1, keepdim=True)[1]
    print([idx2word[idx.item()] for idx in pred.squeeze()])
