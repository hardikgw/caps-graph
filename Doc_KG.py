

class DocEmbedding:
    def __init__(self):
        self.lines_to_process = -1

    def generate_embeddings(self, file_path):
        f = open(file_path)
        for index, line in enumerate(f):
            print(line)
            if index == 0:
                continue
        f.close()


embeddings = DocEmbedding()
embeddings.generate_embeddings('/Users/hardikpatel/workbench/data/patent/text.txt')
