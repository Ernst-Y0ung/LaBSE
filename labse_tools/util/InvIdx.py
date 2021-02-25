import jieba
import pandas as pd
from zhon.hanzi import punctuation


def Tokenizer(corpus_raw, stopwords):
    """
    Tokenize the corpus list
    :param corpus_raw: List of texts that are not preprocessed
    :param stopwords: List of stopwords
    :return: tokenized corpus
    """
    corpus = []
    for i in range(len(corpus_raw)):
        corpus_raw[i] = corpus_raw[i].replace('\n', '')
        seg_list = jieba.cut(corpus_raw[i], cut_all=False)
        token_list = []
        for token in seg_list:
            if not hasNumbers(token) and token not in punctuation and token not in stopwords and len(token) > 2:
                token_list.append(token)
        corpus.append(token_list)
    return corpus


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
