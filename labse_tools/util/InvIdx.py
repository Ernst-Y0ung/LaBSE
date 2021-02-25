import jieba
from zhon.hanzi import punctuation


def tokenize(corpus_raw, stopwords):
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
