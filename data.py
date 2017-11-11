"""
Utility functions
"""
import os
import argparse
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data.vocab import Vocab
import data.vocab as vc

basedir = ""

class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self):
        """
        """  
        self.data_dir="data/validation.p"           # location of reviews data (train|validation)

def sample_data(data_path):
    """
    Sample format of the processed
    data from data.py
    Args:
        data_path: path for train.p|valid.p
    """
    with open(data_path, 'rb') as f:
        entries = pickle.load(f)

    vocab_file = os.path.join(basedir, 'data/vocab.txt')
    vocab = Vocab(vocab_file, verbose=False)

    for k, v in entries.items():
        rand_index = random.randint(0, len(v[0]))
        print ("==> Processed Review:", v[0][rand_index])
        print ("==> Review Len:", v[1][rand_index])
        print ("==> Label:", k)
        print ("==> See if processed review makes sense:",
            vc.ids_to_tokens(
            v[0][rand_index],
            vocab=vocab,
            ))

FLAGS = parameters()
sample_data(FLAGS.data_dir)
