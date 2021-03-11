import jieba
import jieba.posseg as pseg
import re
from zhon.hanzi import punctuation


def hasNumbers(inputString):
    """
    Helper function that checks if a string contains at least one digit
    :param inputString: String to check
    :return: True if the string contains at least one digit, otherwise False
    """
    return any(char.isdigit() for char in inputString)

def tokenize(corpus_raw, stopwords, flags):
    """
    Tokenize the corpus list
    :param corpus_raw: List of texts that are not preprocessed
    :param stopwords: List of stopwords
    :param flags: A list of pos-tags for filtering purpose
    :return: tokenized corpus
    """
    corpus = []
    jieba.enable_paddle()
    for article in corpus_raw:
        words = pseg.cut(article, use_paddle=True)
        token_list = []
        for word, flag in words:
            word = re.sub("[{}]+".format(punctuation), "", word)
            word = word.strip()
            if word not in stopwords and len(word) > 2 and word not in punctuation and flag in flags:
                token_list.append(word)
        corpus.append(token_list)
    return corpus