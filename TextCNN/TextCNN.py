# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])
    targets = []
    for out in labels:
        targets.append(out) # To using Torch Softmax Loss function
    return inputs, targets


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (2, embedding_size)), # => [batch_size, output_channel, 2, 1]
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1)),)
        # fc
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, X):
        '''X: [batch_size, sequence_length]'''
        batch_size = X.shape[0]
        embedding_X = self.W(X) # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X) # [batch_size, output_channel, 1, 1]
        flatten = conved.view(batch_size, -1) # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.FloatTensor

    # 3 words sentences (=sequence_length is 3)
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]    # 1 is good, 0 is not good.

    # TextCNN Parameter
    embedding_size = 2 # wordemb dim
    sequence_length = len(sentences[0]) # every sentences contains sequence_length(=3) words
    num_classes = len(set(labels)) # 0 or 1
    batch_size = 3

    word_list = " ".join(sentences).split()
    vocab = list(set(word_list))
    word2idx = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    max_iter = 2000
    save_interval = int(max_iter * 0.1)
    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    input_batch, target_batch = make_data(sentences, labels)
    input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)

    model = TextCNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training
    for epoch in range(max_iter):
        for idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            if epoch % save_interval == 0:
                model_file = os.path.join(ckpt_path, "textcnn_{}-{}.th".format(epoch, loss.item()))
                print('epoch=', epoch, 'idx=', idx, 'loss=', loss.item(), 'model_file=', model_file)
                torch.save(model.state_dict(), model_file)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    test_text = 'i hate me'
    tests = [[word2idx[n] for n in test_text.split()]]
    test_batch = torch.LongTensor(tests).to(device)

    # Predict
    model = model.eval()
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean")
    else:
        print(test_text, "is Good Mean")
