#coding:utf-8
import gensim
import numpy as np
from numpy import unicode
from sklearn.svm import SVC


if __name__ == '__main__':
    VECTOR_DIR = 'embedding/baike.vectors.bin'
    EMBEDDING_DIM = 200

    print('(1) load texts...')
    train_texts = open('data/train_contents.txt', encoding='utf-8').read().split('\n')
    train_labels = open('data/train_labels.txt', encoding='utf-8').read().split('\n')
    test_texts = open('data/test_contents.txt', encoding='utf-8').read().split('\n')
    test_labels = open('data/test_labels.txt', encoding='utf-8').read().split('\n')


    print('(2) doc to var...')
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
    x_train = []
    x_test = []
    for train_text in train_texts:
        words = train_text.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if unicode(word) in w2v_model:
                vector += w2v_model[unicode(word)]
                word_num += 1
        if word_num > 0:
            vector = vector/word_num
        x_train.append(vector)

    for test_text in test_texts:
        words = test_text.split(' ')
        vector = np.zeros(EMBEDDING_DIM)
        word_num = 0
        for word in words:
            if unicode(word) in w2v_model:
                vector += w2v_model[unicode(word)]
                word_num += 1
        if word_num > 0:
            vector = vector/word_num
        x_test.append(vector)

    y_train = train_labels
    y_test = test_labels
    print('Train doc shape:' + str(len(x_train)) + ', ' + str(len(x_train[0])),
          'Test doc shape:' + str(len(x_test)) + ', ' + str(len(x_test[0])))


    print('(3) SVM...')
    svclf = SVC(kernel='linear')
    svclf.fit(x_train, y_train)
    preds = svclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))
