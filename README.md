## DL-NLP

`DL-NLP` is a tutorial for NLP(Natural Language Processing) based on DL(Deep Learning) by using **Pytorch** and **Tensorflow**.


## Dependencies

- Python 3.8+
- Pytorch 1.6.0+
- Tensorflow 2.0+
- SpaCy
- numpy
- jieba
- gensim
- matplotlib
- sklearn
- pydot
- pydot-ng
- graphviz
- opencc
- sudo apt install python-pydot python-pydot-ng graphviz


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


#### 6. Chinese Word to Vector

- 6.1. 代码位置：Word2Vec/chinese - **中文分词**
- 6.2. 获取中文语料
  - 维基百科的中文语料库质量高、领域广泛而且开放，下载地址https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 
- 6.3. 中文语料库预处理，将xml的Wiki数据转换为text格式
  - python的gensim包中提供了WikiCorpus方法可以直接处理Wiki的语料库（xml的baz格式，无需解压）
  - 执行命令可以将xml的Wiki语料库转换为txt格式：python parse_zhwiki_corpus.py -i zhwiki-latest-pages-articles.xml.bz2 -o corpus.zhwiki.txt
- 6.4. 中文简繁体转换
  - Wiki语料库中的文档含有繁体中文，可以利用工具包opencc将繁体转换为简体
  - 执行命令可以将语料库中的繁体中文转化为简体中文：python chinese_t2s.py -i corpus.zhwiki.txt -o corpus.zhwiki.simplified.txt
- 6.5. 去除英文和空格
  - 现在得到的语料库中有许多英文（也有些许日文、德文等），为避免影响所训练的词向量效果，我们将其中的英文以及空格做了删除（其他日文、德文等后续有时间再进行处理）
  - 执行以下命令可以将语料库中的英文以及空格删除：python remove_en_blank.py -i corpus.zhwiki.simplified.txt -o corpus.zhwiki.simplified.done.txt
- 6.6. 中文分词（jieba分词）
  - 想要完成word2vec的训练，语料库需要进行分词处理，这里采用python的jieba分词
  - 执行以下命令可以将语料库中的中文语料进行分词：python corpus_zhwiki_seg.py -i corpus.zhwiki.simplified.done.txt -o corpus.zhwiki.segwithb.txt
- 6.7. word2vec模型训练
  - python的gensim模块提供了word2vec训练的函数，极大地方便了模型训练的过程
  - 执行以下命令得到所训练的word2vec模型和词向量：python word2vec_train.py -i corpus.zhwiki.segwithb.txt -m zhwiki.word2vec.model -v zhwiki.word2vec.vectors -s 300 -w 5 -n 5
  - 得到基于Wiki中文语料库训练好的word2vec模型和词向量	
  - word2vec模型文件：1)zhwiki.word2vec.model、2)zhwiki.word2vec.model.trainables.syn1neg.npy、3)zhwiki.word2vec.model.wv.vectors.npy	
  - word2vec词向量文件：zhwiki.word2vec.vectors


#### 7. NER (Named Entity Recognition)

- 7.1. NER - **命名实体识别** 序列标注（Sequence Tagging）
- 7.2. NER还在做哪些事情，主要分几个方面
  - 多特征：实体识别不是一个特别复杂的任务，不需要太深入的模型，那么就是加特征，特征越多效果越好，所以字特征、词特征、词性特征、句法特征、KG表征等等的就一个个加，甚至有些中文NER任务里还加入了拼音特征、笔画特征
  - 多任务：很多时候做NER的目的并不仅是为了NER，而是一个大任务下的子任务，比如信息抽取、问答系统等等的，如果要做一个端到端的模型，那么就需要根据自己的需求和场景，做成一个多任务模型，把NER作为其中一个子任务；另外，单独的NER本身也可以做成多任务，比如一个用来识别实体，一个用来判断实体类型
  - 时令大杂烩：把当下比较流行的深度学习话题或方法跟NER结合一下，比如结合强化学习的NER、结合FSL（few-shot learning）的NER、结合多模态信息的NER、结合跨语种学习的NER等等


## Examples

#### 1. Chinese Sentiment Analysis

- 1.1. cn_sentiment_analysis - **中文情感分析**
  - 词向量下载  
    链接: https://pan.baidu.com/s/1GerioMpwj1zmju9NkkrsFg , 提取码: x6v3, 下载之后在项目根目录建立"embeddings"文件夹, 解压放到该文件夹, 即可运行代码


#### 2. Chinese Text Classification

- 2.1. cn_text_classification - **中文文本分类**
  - 基于深度学习方法：CNN、CNN + word2vec、LSTM、LSTM + word2vec、MLP（多层感知机）  
  - 基于传统机器学习方法：朴素贝叶斯、KNN、SVM、SVM + word2vec、SVM + doc2vec
- 2.2. 搜狐新闻数据
  - 下载地址：http://www.sogou.com/labs/resource/cs.php 
- 2.3. word2vec模型文件（使用百度百科文本预训练）
  - 下载：https://pan.baidu.com/s/13QWrN-9aayTTo0KKuAHMhw，提取码 biwh
- 2.4. 实验说明
  - 引入预训练的word2vec 模型会给训练带来好处，具体来说：1、间接引入外部训练数据，防止过拟合；2、减少需要训练的参数个数，提高训练效率
  - LSTM 需要训练的参数个数远小于CNN，但训练时间大于CNN。CNN在分类问题的表现上一直很好，无论是图像还是文本；而想让LSTM优势得到发挥，首先让训练数据量得到保证
  - 将单词在word2vec中的词向量加和求平均获得整个句子的语义向量的方法看似naive有时真挺奏效，当然仅限于短句子，长度100以内应该可以
  - 机器学习方法万千，具体选择用什么样的方法还是要取决于数据集的规模以及问题本身的复杂度，对于复杂程度一般的问题，看似简单的方法有可能是坠吼地


#### 3. Opinion Extraction

- 3.1. opinion_extraction - **英文主题与情感词抽取（细粒度情感分析）**
  - 问题抽象：看作一个类似于分词问题的 “序列标注” 问题，给出分词后的输入序列，输出两个同等长度的BIO序列，一个作为角度词抽取的输出序列结果，一个作为情感词抽取的输出序列结果，这里BIO标记为序列标注问题的惯用标记法，“B” 即为欲标记的目标内容的开始词，“I” 为欲标记内容的中间词（或结尾词），“O” 为不标记的内容
  - 抽取结果：食品质量 → 一般；服务 → 周到，这里 “食品质量” 与 “服务” 是两个不同的角度（aspect，也叫opinion target），前一个角度对应的情感词（opinion word）是 “一般”，极性为负（negative），后一个角度对应的情感词为 “周到”，极性为正（positive）
- 3.2. SemEval2014 (Restaurant)
  - 下载地址：http://alt.qcri.org/semeval2014/task4/
  - 实验使用了Restaurant 的那一部分数据，数据内容是用户在网上对餐厅的评价，从数据示例看，数据集只提供了角度词（aspectTerm）的抽取结果，没有情感词的抽取结果，训练加测试数据总共3841条


## Reference

- Tae Hwan Jung(Jeff Jung) @graykode，modify by [wmathor](https://github.com/wmathor)
