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
