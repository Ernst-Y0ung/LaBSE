import sys
sys.path.append("../")
from labse_tools.util import indexing


if __name__ == '__main__':
    stopwords = open('data/baidu_stopwords.txt', 'r', encoding='utf-8').read().split()
    sentence = ['尽管年初监管层连出政策组合拳，1月人民币兑美元仍升值1.22%录得连八月涨势，人民币CFETS指数也一路攀升至逾两年半新高。']
    flag_need = ['n', 'nr', 'ns', 'nt', 'nw', 'nz', 'PER', 'LOC', 'ORG']
    tokens = indexing.tokenize(sentence, stopwords, flag_need)
    print(tokens)
