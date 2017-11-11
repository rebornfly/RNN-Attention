import os
import ops
import sys
import time
import random
import pickle
basedir = ""
import numpy as np
from data.vocab import Vocab
import tensorflow as tf
from model import Model
from logger import logger
from collections import Counter
from db import DB
import train
import collections

db = DB()
def infer_data(data_path, batch_size):
    print("start infer.....")
    with open(data_path, 'rb') as f:
        entries = pickle.load(f)

    data = []
    for k, v in entries.items():
        if len(v[0]) < batch_size:
            continue
        for i in range(int(len(v[0]) / batch_size)):
            start_index = i*batch_size
            end_index = (i+1)*batch_size
            sub_data = v[0][start_index:end_index]
            sub_lens = v[1][start_index:end_index]
            tmp = [sub_data, sub_lens, k]
            data.append(tmp)

    print(len(data))
    for v in data:

        skuid = [v[2]]
        features = [v[0], skuid]

        yield  features, v[1]

def infer(FLAGS):
    scores = collections.defaultdict(list)
    """
    Infer a previous or new model.
    """
    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/vocab.txt')
    infer_data_path = os.path.join(
        basedir, 'data/infer.p')
    vocab = Vocab(vocab_path)

    # Load embeddings (if using GloVe)
    embeddings = np.zeros((len(vocab), FLAGS.emb_size))
    FLAGS.vocab_size = len(vocab)

    with tf.Session() as sess:
        # Create|reload model
        imdb_model = train.create_model(sess, FLAGS, len(vocab))

        for  infer_index, data in \
            enumerate(infer_data(
                infer_data_path, FLAGS.batch_size)):

            comments, skuid = data[0]
            review_lens = data[1]

            logits,  prob, label= imdb_model.infer(
                sess=sess,
                batch_reviews=comments,
                batch_review_lens=review_lens,
                embeddings=embeddings,
                keep_prob=1.0, # no dropout for val|test
                )
            logger.info ("[INFER]:  [SKUID] : %s | %s | %s",  skuid, label,  prob)
            scores[skuid[0]].append(label[0])

    for k, v in scores.items():
        counts = Counter(v)
        db.update_scores(k, int(counts.most_common(1)[0][0])+5)
        logger.info ("[INFER]:  [SKUID] : %s | %s |%s ",  k, v, counts.most_common(1))

if __name__ == '__main__':
    FLAGS = train.parameters()
    FLAGS.model = 'infer'
    infer(FLAGS)
