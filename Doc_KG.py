class DocEmbedding:
    def __init__(self):
        self.lines_to_process = 10

    def generate_embeddings(self, file_path):
        f = open(file_path)
        for index, line in enumerate(f):
            fields = line.split('\t')
            if index == 0:
                continue
            elif index > self.lines_to_process or self.lines_to_process < 0:
                break
        f.close()


embeddings = DocEmbedding()
embeddings.generate_embeddings('/Users/hardikpatel/workbench/data/patent/brf_sum_text.tsv')
