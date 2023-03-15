"""
Preparation for cutting sentences to words
Read all reviews from sentiment-marked corpus
Write them into one positive and one negative text files
"""
import os
import codecs
import jieba
import jieba.analyse
import re


def getContent(path):
    """
    Transform GBK to UTF-8
    :param path:
    :return: bytes encoded in utf-8
    """
    f = codecs.open(path, 'rb', encoding='GBK', errors='ignore')
    review = f.readline().encode('UTF-8')
    f.close()
    return review


def clearText(content):
    """
    clear number, English & marks
    :param content: uncleared sentences
    :return:
    """
    if content != '':
        content = content.strip().translate(str.maketrans("", ""))
        # rm English & number
        content = re.sub("[a-zA-Z0-9]", "", content)
        # rm marks
        content = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", content)
    return content


def cutSentence(content):
    """
    cut sentence into words
    :param content: str type - sentences
    :return: str type -  words divided by ' '
    """
    segList = jieba.cut(content, cut_all=False)
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + ' '
    return segSentence.strip()


def clearWords(words, stopwords):
    """
     This function is used to remove stopwords which have no effect on sentiment
    :param words: A str - consisted  by divided words
    :param stopwords: A list - stopwords
    :return: A str - words after removing stopwords
    """
    words = words.split(' ')
    words_rm_stopwords = ''
    for word in words:
        if word not in stopwords:
            if word != '\t':
                words_rm_stopwords += word + " "
    return words_rm_stopwords.strip() + '\n'


def prepareData(content):
    return cutSentence(clearText(content)) + '\n'


def prepareReview(sourceFile, targetFile):
    f = codecs.open(sourceFile, 'r', encoding='utf-8')
    target = codecs.open(targetFile, 'w', encoding='utf-8')
    print('open source file: ' + sourceFile)
    print('open target file: ' + targetFile)
    stopwords = [stopword.strip() for stopword in open('/Users/usyun/Documents/CityU/Project/6941-MachineLearning'
                                                       '/sentiment_model/data/stopWord.txt').readlines()]
    lineNum = 1
    line = f.readline()
    while line:
        print('---processing ', lineNum, ' article---')
        processed_line = clearWords(cutSentence(clearText(line)),stopwords)
        target.writelines(processed_line)
        lineNum = lineNum + 1
        line = f.readline()
    print('well done.')
    f.close()
    target.close()


if __name__ == '__main__':
    # Use open source corpus to build review_word_vec model
    corpusDir = './sentiment_model/data/ChnSentiCorp_htl_ba_2000'
    folders = os.listdir(corpusDir)
    for folder in folders:
        output = open('./sentiment_model/data/2000_' + folder + '.txt', 'w')
        files = os.listdir(corpusDir + '/' + folder)
        for file in files:
            filepath = corpusDir + '/' + folder + '/' + file
            words = prepareData(str(getContent(filepath), 'UTF-8'))
            stopwords = [stopword.strip() for stopword in open('./sentiment_model/data/stopWord.txt', 'r').readlines()]
            words_rm_stopwords = clearWords(words, stopwords)
            output.write(words_rm_stopwords)
        output.close()
    # Here input Text form review. One line means one review.
    review_source = '/Users/usyun/Documents/CityU/Project/6941-MachineLearning/sentiment_model/review_data/original_review.txt'
    review_target = '/Users/usyun/Documents/CityU/Project/6941-MachineLearning/sentiment_model/review_data/processed_review.txt'
    prepareReview(review_source,review_target)