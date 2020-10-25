# -*- coding:utf-8 -*-
import os
import random
import time
import gensim
import string
import numpy as np
import torch
import torch.nn as nn	 
import torch.optim as optim
from gensim.models.word2vec import LineSentence
from torch.autograd import Variable
from lstm import LSTMTagger

all_words = {}
wtf_words = {} 


class Config(object):
	data_path = '../../data'
	embedding_path = '../../embedding'

	embedding_dim = 200
	hidden_dim = 80
	batch_size = 1
	label_dim = 3
	learning_rate = 0.1
	drop_rate = 0.5

	max_iter = 100
	statistic_step = int(max_iter * 0.1)
	test_batch_size = 128

	ckpt_path = 'ckpts'
	if not os.path.exists(ckpt_path):
		os.makedirs(ckpt_path)


def load_data_to_vecs(path, embedding_file, style):
	if style=='x':
		model = gensim.models.Word2Vec.load(embedding_file)
		lines = open(path).read().split('\n')
		vecs = []

		for line in lines:
			line = line.translate(string.punctuation)
			words = line.split(' ')
			line_vecs = []
			for word in words:
				if word not in all_words:
					all_words[word] = ''
				if word in model:
					line_vecs.append(model[word].tolist())
				else:
					line_vecs.append([0]*200)
					if word not in wtf_words:
						wtf_words[word] = ''
			vecs.append(line_vecs)
		return vecs

	if style=='y':
		lines = open(path).read().split('\n')
		vecs = []
		for line in lines:
			labels = line.split(' ')
			line_vecs = [int(label) for label in labels]
			vecs.append(line_vecs)
		return vecs


def train_d2v_model(infile, embedding_file):
	model = gensim.models.Word2Vec(LineSentence(infile), size=200, window=5, min_count=5)
	model.save(embedding_file)


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config = Config()

	print('(1) load data and trans to vecs...')
	embedding_file = os.path.join(config.embedding_path, 'yelp.vector.bin')
	if not os.path.exists(embedding_file):
		train_d2v_model(os.path.join(config.data_path, 'train_docs.txt'), embedding_file)
		print('embedding_file not exist and then trained')

	x_train = load_data_to_vecs(os.path.join(config.data_path, 'train_docs.txt'), embedding_file, style='x')
	y_train = load_data_to_vecs(os.path.join(config.data_path, 'train_labels_a.txt'), embedding_file, style='y')
	print('there are ' + str(len(all_words)) + ' words totally')
	print('there are ' + str(len(wtf_words)) + ' words not be embeded')
	print('train docs:' + str(len(x_train)))
	print('train labels of aspect:' + str(len(y_train)))


	print('(2) build model...')
	model = LSTMTagger(config)
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
	print(model)
	

	print('(3) train model...')
	start = time.time()
	for epoch in range(config.max_iter):
		less_loss = 10000.0
		print('epoch=', epoch, 'x_train.len=', len(x_train))

		for i in range(len(x_train)):
			model.zero_grad()  # 清除网络先前的梯度值，梯度值是Pytorch的变量才有的数据，Pytorch张量没有
			model.hidden = model.init_hidden()  # 重新初始化隐藏层数据，避免受之前运行代码的干扰
			out = model(Variable(torch.FloatTensor(x_train[i])))
			loss = loss_function(out, Variable(torch.LongTensor(y_train[i])))
			loss.backward()
			optimizer.step()

			loss_value = loss.data.numpy()
			#print('epoch:' + str(epoch) + ' / ' + str(config.max_iter), 'steps:' + str(i),
			#      'loss:', str(loss_value), 'cost_time:' + str(time.time() - start))

			if loss_value < less_loss:
				model_file = os.path.join(config.ckpt_path, "lstm_pth_{}-{}-{}.th".format(epoch, i, loss_value))
				torch.save(model.state_dict(), model_file)
				print('epoch:' + str(epoch), 'steps:' + str(i), 'model_file=', model_file)
				less_loss = loss_value
				hits = 0

			if i % config.statistic_step == 0:
				for j in range(config.test_batch_size):
					index = random.randint(0, len(x_train) - 1)
					out = model(Variable(torch.FloatTensor(x_train[j])))
					out = out.data.numpy()
					if all(np.argmax(out[index]) == y_train[j][index] for index in range(len(y_train[j]))):
						hits += 1
				print('epoch:' + str(epoch), 'steps:' + str(i), 'precision: ' + str(float(hits) / config.test_batch_size))
