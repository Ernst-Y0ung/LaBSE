class Indexing:
    def __init__(self, corpus):
        self.corpus = corpus
        self.adj_mat = []
        self.mapper_inv = []
        self.inv_idx = {}
        self.inv_idx_count = {}

    def build_inv_index(self):
        """
        Build the inverted index of the corpus
        """
        for i in range(len(self.corpus)):
            for word in self.corpus[i]:
                if word not in self.inv_idx:
                    self.inv_idx[word] = set()
                self.inv_idx[word].add(i)

        for key, value in self.inv_idx.items():
            self.inv_idx_count[key] = len(value)
        self.inv_idx_count = dict(sorted(self.inv_idx_count.items(), reverse=True, key=lambda item: item[1]))
        return

    def build_adjacency_matrix(self):
        """
        Build the adjacency matrix based on corpus given
        :return: An adjacency matrix and a mapper containing every keyword and its numeric index
        """
        if len(self.inv_idx) == 0:
            print('Please run build_inv_index first')
            return
        count = len(self.inv_idx)
        mapper = {}
        self.adj_mat = [[0 for _ in range(count)] for _ in range(count)]

        for idx, (item, value) in enumerate(self.inv_idx.items()):
            mapper[item] = idx
            self.mapper_inv.append(item)

        for idx, (item, value) in enumerate(self.inv_idx.items()):
            from_idx = idx
            for article in value:
                for token in self.corpus[article]:
                    to_idx = mapper[token]
                    self.adj_mat[from_idx][to_idx] += 1

        assert len(self.adj_mat) == len(self.adj_mat[0])
        return self.adj_mat, self.mapper_inv

    def dump_adjacency_matrix(self, outfile):
        """
        Dump an adjacency matrix to a file
        :param outfile: Path of the output file
        :return:
        """
        with open(outfile, 'a', encoding='utf-8') as file:
            for idx, line in enumerate(self.adj_mat):
                token = self.mapper_inv[idx]
                line = '\t'.join([str(e) for e in line])
                # print(line)
                file.write(token + '\t' + line + '\n')
