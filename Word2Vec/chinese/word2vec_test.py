from gensim.models import Word2Vec


if __name__ == '__main__':
    WORD2VEC_MODEL_DIR = 'model/zhwiki.word2vec.model'
    word2vec_model = Word2Vec.load(WORD2VEC_MODEL_DIR)

    # 查看词向量
    vec = word2vec_model['北京']
    print('北京：', vec.shape)

    # 查看相似词
    sim_words = word2vec_model.most_similar('北京')
    print('The most similar words: ')
    for idx, w in enumerate(sim_words):
        print('idx=', idx, 'similar word=', w)
