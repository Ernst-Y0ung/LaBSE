import jieba
import jieba.posseg as pseg
import re
from zhon.hanzi import punctuation


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


def build_invidx(corpus):
    """
    Build the inverted index of the corpus
    :param corpus: List of tokenized corpus
    :return: inv_idx: The inverted index; inv_idx_count: Counter
    """
    inv_idx = {}
    inv_idx_count = {}

    for i in range(len(corpus)):
        for word in corpus[i]:
            if word not in inv_idx:
                inv_idx[word] = set()
            inv_idx[word].add(i)

    for key, value in inv_idx.items():
        inv_idx_count[key] = len(value)
    inv_idx_count = dict(sorted(inv_idx_count.items(), reverse=True, key=lambda item: item[1]))

    return inv_idx, inv_idx_count

def hasNumbers(inputString):
    """
    Helper function that checks if a string contains at least one digit
    :param inputString: String to check
    :return: True if the string contains at least one digit, otherwise False
    """
    return any(char.isdigit() for char in inputString)
