import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import tensorflow_hub as hub
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 200)
from sklearn.metrics.pairwise import cosine_similarity

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


class DocEmbedding:
    def __init__(self):
        self.lines_to_process = 10

    def generate_embeddings(self, file_path):
        f = open(file_path)
        for index, line in enumerate(f):
            fields = line.split('\t')
            print(fields)
            if index == 0:
                continue
            elif index > self.lines_to_process or self.lines_to_process < 0:
                break
        f.close()

    def elmo_vectors(x):
        embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))

        elmo_embeddings = []
        print(len(corpus))
        for i in range(len(corpus)):
            print(corpus[i])
            elmo_embeddings.append(elmo_vectors([corpus[i]])[0])
        print(elmo_embeddings)
        print(cosine_similarity(elmo_embeddings, elmo_embeddings))


doc_embeddings = DocEmbedding()
doc_embeddings.generate_embeddings('/Users/hardikpatel/workbench/data/patent/brf_sum_text.tsv')
