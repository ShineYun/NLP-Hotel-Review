# 从词向量模型中提取文本特征向量
import warnings
import logging
import os.path
import codecs, sys
import numpy as np
import pandas as pd
import gensim


def getWordVecs(wordList, model):
    vecs = []

    for word in wordList:
        word = word.replace('\n', '')
        # print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')


# 构建文档词向量
def buildVecs(filename, model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList, model)
            # print vecs
            # sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) > 0:
                vecsArray = sum(np.array(vecs)) / len(vecs)  # mean
                # print vecsArray
                # sys.exit()
                fileVecs.append(vecsArray)
            else:
                fileVecs.append(np.zeros(400))
    return fileVecs


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # load word2vec model
    word2vec_path = './wiki_zh_word2vec/wiki_zh_vec/wiki.zh.text.vector'
    logger.info("Loading word2vec model...")
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    logger.info("Load over")

    logger.info("Building word vectors for review...")
    marked_clean_data_path = './sentiment_model/data'
    posInput = buildVecs(marked_clean_data_path + '/2000_pos.txt', model)
    logger.info('PosInput Length:'+str(len(posInput)))
    negInput = buildVecs(marked_clean_data_path + '/2000_neg.txt', model)
    logger.info('NegInput Length:'+str(len(posInput)))
    # Before building model, review data should also be cleaned
    test = buildVecs('sentiment_model/review_data/processed_review.txt', model)
    logger.info('Test Length:'+str(len(posInput)))
    logger.info("Build over")

    df_test = pd.DataFrame(np.array(test[:]))
    df_test.to_csv('review_WordVec_data.csv')

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y, df_x], axis=1)
    # print data
    data.to_csv('2000_data.csv')
    print('over')
