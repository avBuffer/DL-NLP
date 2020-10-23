# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data


def make_data(sentences):
  inputs = []
  for sen in sentences:
      inputs.append(np.asarray([word2idx[n] for n in sen.split()]))
  targets = []
  for out in labels:
      targets.append(out) # To using Torch Softmax Loss function
  return torch.LongTensor(inputs), torch.LongTensor(targets)


class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X):
        '''X: [batch_size, seq_len]'''
        input = self.embedding(X) # input : [batch_size, seq_len, embedding_dim]
        input = input.transpose(0, 1) # input : [seq_len, batch_size, embedding_dim]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1) # output : [batch_size, seq_len, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Bi-LSTM(Attention) Parameters
    batch_size = 3
    embedding_dim = 2
    n_hidden = 5 # number of hidden units in one cell
    num_classes = 2  # 0 or 1

    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    vocab = list(set(" ".join(sentences).split()))
    word2idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(word2idx)

    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    inputs, targets = make_data(sentences)
    dataset = Data.TensorDataset(inputs, targets)
    loader = Data.DataLoader(dataset, batch_size, True)

    model = BiLSTM_Attention().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(max_iter):
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            pred, attention = model(x)
            loss = criterion(pred, y)
            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "bi-lstm_attention_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    test_text = 'i hate me'
    tests = [np.asarray([word2idx[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests).to(device)

    # Predict
    predict, _ = model(test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean")
    else:
        print(test_text, "is Good Mean")

    fig = plt.figure(figsize=(10, 10)) # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention.cpu().data, cmap='viridis')
    ax.set_xticklabels([''] + ['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
    #plt.show()
    plt.savefig(os.path.join(ckpt_path, 'Bi-LSTM_Attention.png'))
