# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data


def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (n_step - len(seq[i]))  # 'man??', 'women'

        enc_input = [letter2idx[n] for n in (seq[0] + 'E')]  # ['m', 'a', 'n', '?', '?', 'E']
        dec_input = [letter2idx[n] for n in ('S' + seq[1])]  # ['S', 'w', 'o', 'm', 'e', 'n']
        dec_output = [letter2idx[n] for n in (seq[1] + 'E')]  # ['w', 'o', 'm', 'e', 'n', 'E']

        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)  # not one-hot
    # make tensor
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.LongTensor(dec_output_all)


class TranslateDataSet(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self):  # return dataset size
        return len(self.enc_input_all)

    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]


# Model
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # encoder
        self.decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)  # decoder
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input(=input_batch): [batch_size, n_step+1, n_class]
        # dec_inpu(=output_batch): [batch_size, n_step+1, n_class]
        enc_input = enc_input.transpose(0, 1)  # enc_input: [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)  # dec_input: [n_step+1, batch_size, n_class]

        # h_t : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, h_t = self.encoder(enc_input, enc_hidden)
        # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.decoder(dec_input, h_t)
        model = self.fc(outputs)  # model : [n_step+1, batch_size, n_class]
        return model


def translate(word):
    enc_input, dec_input, _ = make_data([[word, '?' * n_step]])
    enc_input, dec_input = enc_input.to(device), dec_input.to(device)
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden).to(device)
    output = model(enc_input, hidden, dec_input)
    # output : [n_step+1, batch_size, n_class]

    predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
    decoded = [letter[i] for i in predict]
    translated = ''.join(decoded[:decoded.index('E')])
    return translated.replace('?', '')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # S: Symbol that shows starting of decoding input
    # E: Symbol that shows starting of decoding output
    # ?: Symbol that will fill in blank sequence if current batch data size is short than n_step

    letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
    letter2idx = {n: i for i, n in enumerate(letter)}
    n_class = len(letter2idx) # classfication problem

    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    # Seq2Seq Parameter   
    batch_size = 3
    n_step = max([max(len(i), len(j)) for i, j in seq_data]) # max_len(=5)
    n_hidden = 128

    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    '''
    enc_input_all: [6, n_step+1 (because of 'E'), n_class]
    dec_input_all: [6, n_step+1 (because of 'S'), n_class]
    dec_output_all: [6, n_step+1 (because of 'E')]
    '''
    enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)
    loader = Data.DataLoader(TranslateDataSet(enc_input_all, dec_input_all, dec_output_all), batch_size, True)

    model = Seq2Seq().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(max_iter):
        for idx, (enc_input_batch, dec_input_batch, dec_output_batch) in enumerate(loader):
            # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
            h_0 = torch.zeros(1, batch_size, n_hidden).to(device)
            enc_input_batch, dec_intput_batch, dec_output_batch = enc_input_batch.to(device), dec_input_batch.to(device), dec_output_batch.to(device)          # enc_input_batch : [batch_size, n_step+1, n_class]

            # dec_intput_batch : [batch_size, n_step+1, n_class]
            # dec_output_batch : [batch_size, n_step+1], not one-hot
            pred = model(enc_input_batch, h_0, dec_intput_batch)
            # pred : [n_step+1, batch_size, n_class]
            pred = pred.transpose(0, 1) # [batch_size, n_step+1(=6), n_class]

            loss = 0
            for i in range(len(dec_output_batch)):
                # pred[i] : [n_step+1, n_class]
                # dec_output_batch[i] : [n_step+1]
                loss += criterion(pred[i], dec_output_batch[i])

            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "textlstm_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    print('Test')
    print('man ->', translate('man'))
    print('men ->', translate('men'))
    print('woman ->', translate('woman'))
    print('women ->', translate('women'))

    print('black ->', translate('black'))
    print('white ->', translate('white'))

    print('king ->', translate('king'))
    print('queen ->', translate('queen'))

    print('girl ->', translate('girl'))
    print('girls ->', translate('girls'))
    print('boy ->', translate('boy'))
    print('boys ->', translate('boys'))

    print('up ->', translate('up'))
    print('down ->', translate('down'))

    print('high ->', translate('high'))
    print('low ->', translate('low'))
