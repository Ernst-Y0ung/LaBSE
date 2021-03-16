import sys
sys.path.append("../")
from labse_tools.util.indexing import Indexing
from labse_tools.util.preprocess import tokenize
import pandas as pd


if __name__ == '__main__':
    stopwords = open('data/baidu_stopwords.txt', 'r', encoding='utf-8').read().split()
    # sentence = ['尽管年初监管层连出政策组合拳，1月人民币兑美元仍升值1.22%录得连八月涨势，人民币CFETS指数也一路攀升至逾两年半新高。']
    df = pd.read_json('E:/Github/similarity visualization/data/Data.json', orient='index')
    corpus_raw = df['Text'].tolist()

    # Tokenizing
    flag_need = ['n', 'nr', 'ns', 'nt', 'nw', 'nz', 'PER', 'LOC', 'ORG']
    corpus = tokenize(corpus_raw, stopwords, flag_need)

    # Indexing
    idx = Indexing(corpus)
    idx.build_inv_index()
    idx.build_adjacency_matrix()
    print(idx.inv_idx)
    # print(idx.adj_mat)
