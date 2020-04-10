import tensorflow as tf
import os
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split

df_data = pd.read_csv("data/ner/ner_dataset.csv", sep=",", encoding="latin1").fillna(method='ffill')
print(df_data.shape)


x_train, x_test = train_test_split(df_data, test_size=0.20, shuffle=False)

agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                   s["POS"].values.tolist(),
                                                   s["Tag"].values.tolist())]

x_train_grouped = x_train.groupby("Sentence #").apply(agg_func)
x_test_grouped = x_test.groupby("Sentence #").apply(agg_func)

MAX_LENGTH = 128
BERT_MODEL = "bert-base-cased"

BATCH_SIZE = 32

pad_token = 0
pad_token_segment_id = 0
sequence_a_segment_id = 0

from transformers import (
    TF2_WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    TFBertForTokenClassification,
    create_optimizer)

MODEL_CLASSES = {"bert": (BertConfig, TFBertForTokenClassification, BertTokenizer)}

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)