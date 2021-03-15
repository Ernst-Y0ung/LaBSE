import numpy as np
import operator
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min


def print_list(arr):
    """
    Helper function
    :param arr: A list
    :return: None
    """
    for item in arr:
        print(item)


def calculate_mmr(n, lamb, corpus, corpus_embedding):
    """
    Maximal Marginal Relevance
    :param n: Number of output sentences
    :param lamb: Lambda in the MMR formula
    :param corpus: List of texts
    :param corpus_embedding: List of the texts' LaBSE embedding
    :return: The summary text list
    """
    result = np.array([])
    cor_len = len(corpus)
    while n > 0:
        mmr = {}
        for s in range(cor_len):
            if s not in mmr:
                # Calculate sim1
                sum_arr = sum([item for item in corpus_embedding])
                sim1 = cosine_similarity(corpus_embedding[s].reshape(1, -1), sum_arr.reshape(1, -1))

                # Calculate sim2
                if len(result) == 0:
                    sim2 = 0
                else:
                    summary_arr = sum([corpus_embedding[int(x)] for x in result])
                    sim2 = cosine_similarity(corpus_embedding[s].reshape(1, -1), summary_arr.reshape(1, -1))
                mmr[s] = lamb * sim1 - (1 - lamb) * sim2

        # Add the max value to the result
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        result = np.append(result, selected)
        n -= 1

    res_idx = sorted([int(x) for x in result.tolist()])
    return [corpus[x] for x in res_idx]


def clustering(corpus, corpus_embedding):
    """
    Clustering using k-means
    :param corpus: List of texts
    :param corpus_embedding: List of the texts' LaBSE embedding
    :return: The summary text list
    """
    n_clusters = int(np.ceil(len(corpus_embedding) ** 0.5))

    model = KMeans(n_clusters=n_clusters)
    model.fit(corpus_embedding)

    avg = []
    for i in range(n_clusters):
        idx = np.where(model.labels_ == i)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, corpus_embedding)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])

    summary = [corpus[closest[idx]] for idx in ordering]
    return summary
