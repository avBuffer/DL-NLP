import os
import pickle
import random
import numpy as np

## tags, BIO
tag2label = {"O": 0, "B-DISEASE": 1, "I-DISEASE": 2, "B-SYMPTOM": 3, "I-SYMPTOM": 4, "B-BODY": 5, "I-BODY": 6}


def read_corpus(corpus_path):
    """read corpus and return the list of samples"""
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n' and len(line.strip().split()) == 2:
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


def build_vocab(vocab_path, corpus_path, min_count):
    """BUG: I forget to transform all the English characters from full-width into half-width..."""
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1

    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1

    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def create_voabulary(cache_path='word_voabulary.pkl', word_file='words.txt'):
    # load the cache file if exists
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            vocabulary_word2index, vocabulary_index2word = pickle.load(data_f)
            return vocabulary_word2index, vocabulary_index2word
    else:
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        words = open(word_file).readlines()
        print('vocabulary:', len(words))
        for i, vocab in enumerate(words):
            vocabulary_word2index[vocab] = i + 1
            vocabulary_index2word[i + 1] = vocab

        # save to file system if vocabulary of words is not exists.
        print(len(vocabulary_word2index))
        if not os.path.exists(cache_path):
            with open(cache_path, 'wb') as data_f:
                pickle.dump((vocabulary_word2index, vocabulary_index2word), data_f)
    return vocabulary_word2index, vocabulary_index2word
