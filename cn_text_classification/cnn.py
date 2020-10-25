#coding:utf-8
import os
import numpy as np
from tensorflow.python.keras.layers import Embedding, Dropout, Conv1D, Flatten, Dense, MaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model


if __name__ == '__main__':
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 200
    VALIDATION_SPLIT = 0.16
    TEST_SPLIT = 0.2

    ckpt_path = 'ckpts'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    print('(1) load texts...')
    train_texts = open('data/train_contents.txt', encoding='utf-8').read().split('\n')
    train_labels = open('data/train_labels.txt', encoding='utf-8').read().split('\n')
    test_texts = open('data/test_contents.txt', encoding='utf-8').read().split('\n')
    test_labels = open('data/test_labels.txt', encoding='utf-8').read().split('\n')
    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels


    print('(2) doc to var...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(all_labels))
    print('Shape of data tensor:', data.shape, 'Shape of label tensor:', labels.shape)


    print('(3) split data set...')
    # split the data into training set, validation set, and test set
    p1 = int(len(data) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    p2 = int(len(data) * (1 - TEST_SPLIT))
    x_train = data[:p1]
    y_train = labels[:p1]
    x_val = data[p1:p2]
    y_val = labels[p1:p2]
    x_test = data[p2:]
    y_test = labels[p2:]
    print('train docs: ' + str(len(x_train)), 'val docs: ' + str(len(x_val)), 'test docs: ' + str(len(x_test)))


    print('(4) training model...')
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.2))
    model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()
    plot_model(model, to_file=os.path.join(ckpt_path, 'cnn_model.png'), show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    print('model.metrics_names=', model.metrics_names)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
    model.save(os.path.join(ckpt_path, 'cnn.h5'))


    print('(5) testing model...')
    print(model.evaluate(x_test, y_test))
