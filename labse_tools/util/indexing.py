def build_inv_index(corpus):
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

def build_adjacency_matrix(corpus, inv_idx):
    """
    Build the adjacency matrix based on corpus given
    :param corpus: A list containing list of keywords (for each articles)
    :param inv_idx: The inverted index of the corpus
    :return: An adjacency matrix and a mapper containing every keyword and its numeric index
    """
    count = len(inv_idx)
    mapper = {}
    mapper_inv = []
    adj_mat = [[0 for _ in range(count)] for _ in range(count)]

    for idx, (item, value) in enumerate(inv_idx.items()):
        mapper[item] = idx
        mapper_inv.append(item)

    for idx, (item, value) in enumerate(inv_idx.items()):
        from_idx = idx
        for article in value:
            for token in corpus[article]:
                to_idx = mapper[token]
                adj_mat[from_idx][to_idx] += 1

    assert len(adj_mat) == len(adj_mat[0])
    return adj_mat, mapper_inv

def dump_adjacency_matrix(outfile, adj_mat, mapper_inv):
    """
    Dump an adjacency matrix to a file
    :param outfile: Path of the output file
    :param adj_mat: The adjacency matrix to be dumped
    :param mapper_inv: A mapper containing every keyword from adj_mat and its numeric index
    :return:
    """
    with open(outfile, 'a', encoding='utf-8') as file:
        for idx, line in enumerate(adj_mat):
            token = mapper_inv[idx]
            line = '\t'.join([str(e) for e in line])
            # print(line)
            file.write(token + '\t' + line + '\n')
