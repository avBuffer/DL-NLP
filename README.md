## DL-NLP

`DL-NLP` is a tutorial for NLP(Natural Language Processing) based on DL(Deep Learning) by using **Pytorch** and **Tensorflow**.


## Dependencies

- Python 3.8+
- Pytorch 1.6.0+
- Tensorflow 2.0+
- SpaCy


## Models

#### 0. NLP Mind Mapping

- 0.1. **基于深度学习NLP成长之路** in doc folder


#### 1. Basic Embedding Model

- 1.1. NNLM(Neural Network Language Model) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 1.2. Word2Vec(Skip-gram) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- 1.3. FastText(Application Level) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)


#### 2. CNN (Convolutional Neural Network)

- 2.1. TextCNN - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)


#### 3. RNN (Recurrent Neural Network)

- 3.1. TextRNN - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- 3.2. TextLSTM - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 3.3. Bi-LSTM - **Predict Next Word in Long Sentence**
  - Colab -  [Bi_LSTM_Torch.ipynb](https://colab.research.google.com/drive/1R_3_tk-AJ4kYzxv8xg3AO9rp7v6EO-1n?usp=sharing)


#### 4. Attention Mechanism

- 4.1. Seq2Seq - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
- 4.2. Seq2Seq_Attention - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
- 4.3. Bi-LSTM_Attention - **Binary Sentiment Classification**


#### 5. Model based on Transformer

- 5.1. Transformer - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- 5.2. BERT - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)



## Reference

- Tae Hwan Jung(Jeff Jung) @graykode，modify by [wmathor](https://github.com/wmathor)
