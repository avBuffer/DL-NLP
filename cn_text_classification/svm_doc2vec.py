#coding:utf-8
import os
import gensim
from sklearn.svm import SVC


def train_d2v_model(all_texts_file, DOC2VEC_DIR):
    sentences = gensim.models.doc2vec.TaggedLineDocument(all_texts_file)
    model = gensim.models.Doc2Vec(sentences, size=200, window=5, min_count=5)
    model.save(DOC2VEC_DIR)
    print('num of docs:' + str(len(model.docvecs)))


if __name__ == '__main__':
    DOC2VEC_DIR = 'embedding/doc2vec.bin'

    print('(1) load texts...')
    train_texts = open('data/train_contents.txt', encoding='utf-8').read().split('\n')
    train_labels = open('data/train_labels.txt', encoding='utf-8').read().split('\n')
    test_texts = open('data/test_contents.txt', encoding='utf-8').read().split('\n')
    test_labels = open('data/test_labels.txt', encoding='utf-8').read().split('\n')


    print('(2.1) training doc2vec model...')
    if not os.path.exists(DOC2VEC_DIR):
        all_texts = train_texts + test_texts
        all_texts_file = 'data/all_contents.txt'
        fout = open(all_texts_file, 'w', encoding='utf-8')
        fout.write('\n'.join(all_texts))
        fout.close()
        train_d2v_model(all_texts_file, DOC2VEC_DIR)

    print('(2.2) load doc2vec model...')
    model = gensim.models.Doc2Vec.load(DOC2VEC_DIR)
    x_train = []
    x_test = []
    y_train = train_labels
    y_test = test_labels
    for idx, docvec in enumerate(model.docvecs.vectors_docs):
        print('idx=', idx, 'docvec.shape=', docvec.shape)
        if idx < 17600:
            x_train.append(docvec)
        else:
            x_test.append(docvec)
    print('Train doc shape:' + str(len(x_train)) + ', ' + str(len(x_train[0])),
          'Test doc shape:' + str(len(x_test)) + ', ' + str(len(x_test[0])))


    print('(3) SVM...')
    svclf = SVC(kernel='rbf')
    svclf.fit(x_train, y_train)
    preds = svclf.predict(x_test)
    num = 0
    preds = preds.tolist()
    for i, pred in enumerate(preds):
        if int(pred) == int(y_test[i]):
            num += 1
    print('precision_score:' + str(float(num) / len(preds)))
