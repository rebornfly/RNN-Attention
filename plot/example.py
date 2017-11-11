import os
import sys
import time
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体
sys.path.append('..')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from data.vocab import Vocab
from data import vocab as vc
import sys

import matplotlib
matplotlib.use('Agg')
basedir = ""

max_length = 40

myfont = fm.FontProperties(fname='/usr/share/fonts/msyh.ttf')
print(fm.FontProperties().get_family())
from tqdm import (
    tqdm,
)
class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self):
        """
        """
        self.data_dir=""           # location of reviews data
        self.ckpt_dir="../data/ckpt"      # location of model checkpoints
        self.model_name="imdb_model"                     # Name of the model
        self.sample_num=3                                # Sample num to view attn plot. [0-4]
        self.num_rows=1                                  # Number of rows to show in attn visualization.

def plot_cmt_attn(attn_cmt):
    """
    Plot the attention scores.
    Args:
        input_sentence: input sentence (tokens) without <pad>
        attentions: attention scores for each token in input_sentence
        num_rows: how many rows you want the figure to have (we will add 1)
        save_loc: fig will be saved to this location
    """

    # Determine how many words per row
    num_rows = 40
    words_per_row = (len(attn_cmt[0])//num_rows)
    print("comment attention ==========> ", words_per_row, attn_cmt)
    #print(".....word....",row_num, num_rows, words_per_row)

    # Use one extra row in case of remained for quotient above
    fig, axes = plt.subplots(nrows=num_rows+1, ncols=1, figsize=(20, 40))
    for row_num, ax in enumerate(axes.flat):

        # Isolate pertinent part of sentence and attention scores
        start_index = row_num*words_per_row
        end_index = (row_num*words_per_row)+words_per_row
        _attentions = np.reshape(
            attn_cmt[0, start_index:end_index],
            (1, len(attn_cmt[0, start_index:end_index]))
        )

        print("find...........",_attentions)
        # Plot attn scores (constrained to [0.9, 1] for emphasis)
        im = ax.imshow(_attentions, cmap='Blues' )

        # Set up axes
        _input_sentence = list(range(start_index, end_index))
        print(row_num, start_index, end_index, _input_sentence)
        ax.set_xticklabels(
            [''] + _input_sentence,
            minor=False,
            )
        ax.set_yticklabels([''])

        # Set x tick to top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='black')

        # Show corresponding words at the ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Add color bar
    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])

    # display color bar
    cb = fig.colorbar(im, cax=cbar)
    cb.set_ticks([]) # clean color bar

    fig.savefig("image/cmt.pdf", dpi=fig.dpi, bbox_inches='tight') # dpi=fig.dpi for high res. save
def plot_word_attn(attn_word):

    # Determine how many words per row
    #words_per_row = (len(input_sentence.split(' '))//num_rows)
    num_rows = 1
    save_loc = "image/example.pdf"
    fig, axes = plt.subplots(nrows=len(attn_word)+1, ncols=1, figsize=(20, len(attn_word)*2 ))
    for i, attn in enumerate(attn_word):

        ax = axes.flat[i]

        sample = "sample_%i" %i
        review_len = attn_word[sample]["review_len"]
        review = attn_word[sample]["review"]
        label = attn_word[sample]["label"]
        attn_scores = attn_word[sample]["attn_scores"]

        input_sentence, plot_attn_scores = get_attn_inputs(
            FLAGS=FLAGS,
            review=review,
            review_len=review_len,
            raw_attn_scores=attn_scores,
        )

        attentions=plot_attn_scores
        words_per_row = (len(input_sentence)//num_rows)

        #end_index = (row_num*words_per_row)+words_per_row
        end_index = words_per_row + 1
        _input_sentence = input_sentence[0:end_index]
        _attentions = np.reshape(
            attentions[0, 0:end_index],
            (1, len(attentions[0, 0:end_index]))
        )

        # Plot attn scores (constrained to [0.9, 1] for emphasis)
        im = ax.imshow(_attentions, cmap='Blues')

        # Set up axes
        #_input_sentence = 'fjdksljfkdlkffdsfsfdsf'
        ax.set_xticklabels(
            [' ']+list(_input_sentence),
            fontsize=16,
            )
        ax.set_xlabel(i)
        ax.set_yticklabels([str(i)])

        # Set x tick to top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='black')

        # Show corresponding words at the ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Add color bar
    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])

    # display color bar
    cb = fig.colorbar(im, cax=cbar)
    cb.set_ticks([]) # clean color bar

    if save_loc is None:
        # Show the plot
        plt.show()
    else:
        # Save the plot
        fig.savefig(save_loc, dpi=fig.dpi, bbox_inches='tight') # dpi=fig.dpi for high res. save

def get_attn_inputs(FLAGS, review, review_len, raw_attn_scores):
    """
    Return the inputs needed to
    plot the attn scores. These include
    input_sentence and attn_scores.
    Args:
        FLAGS: parameters
        review: list of ids
        review_len: len of the relevant review
    Return:
        input_sentence: inputs as tokens (words) on len <review_len>
        plot_attn_scoes: (1, review_len) shaped scores
    """
    # Data paths
    vocab_path = os.path.join(
        basedir, '../data/vocab.txt')
    vocab = Vocab(vocab_path)

    review = review[:review_len]
    attn_scores = raw_attn_scores[:review_len]
    attn_scores = raw_attn_scores

    # Process input_sentence
    _input_sentence = vc.ids_to_tokens(review, vocab)
    _input_sentence += ['.']*(max_length-len(_input_sentence))
    input_sentence = ''.join(item for item in _input_sentence)

    print("plot ...........", input_sentence)
    print("plot ...........", attn_scores)
    # Process attn scores (normalize scores between [0,1])
    min_attn_score = min(attn_scores)
    max_attn_score = max(attn_scores)
    normalized_attn_scores = ((attn_scores - min_attn_score) / \
        (max_attn_score - min_attn_score))

    # Reshape attn scores for plotting
    plot_attn_scores = np.zeros((1, max_length))
    for i, score in enumerate(normalized_attn_scores):
        plot_attn_scores[0, i] = score

    #print(plot_attn_scores)
    return input_sentence, plot_attn_scores

def process_sample_attn(FLAGS):
    """
    Use plot_attn from utils.py to visualize
    the attention scores for a particular
    sample FLAGS.sample_num.
    """

    # Load the attn history
    attn_word_path = os.path.join(
        basedir, FLAGS.ckpt_dir, 'attn_word_history.p')
    with open(attn_word_path, 'rb') as f:
        attn_word = pickle.load(f)

    attn_cmt_path = os.path.join(
        basedir, FLAGS.ckpt_dir, 'attn_cmt_history.p')
    with open(attn_cmt_path, 'rb') as f:
        attn_cmt = pickle.load(f)

    #print(attn_word, attn_cmt)
    plot_word_attn(attn_word)
    plot_cmt_attn(attn_cmt)

FLAGS = parameters()
process_sample_attn(FLAGS)
