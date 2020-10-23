#!/usr/bin/env python
# coding: utf-8

# ## Install [FastText](https://fasttext.cc/docs/en/supervised-tutorial.html)
#get_ipython().system('wget https://github.com/facebookresearch/fastText/archive/0.2.0.zip')
#get_ipython().system('unzip 0.2.0.zip')
#get_ipython().run_line_magic('cd', 'fastText-0.2.0')
#get_ipython().system('make')

# ## Make simple dataset
# 1 is positive, 0 is negative
f = open('train.txt', 'w')
f.write('__label__1 i love you\n')
f.write('__label__1 he loves me\n')
f.write('__label__1 she likes baseball\n')
f.write('__label__0 i hate you\n')
f.write('__label__0 sorry for that\n')
f.write('__label__0 this is awful')
f.close()

f = open('test.txt', 'w')
f.write('sorry hate you')
f.close()

# ## Training
#get_ipython().system('./fasttext supervised -input train.txt -output model -dim 2')

# ## Predict
#get_ipython().system('cat test.txt')
#get_ipython().system('./fasttext predict model.bin test.txt')
