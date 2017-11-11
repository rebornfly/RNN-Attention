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
import data.vocab as vc

def infer_data(data_path, batch_size):
    with open(data_path, 'rb') as f:
        entries = pickle.load(f)

    data = []
    for k, v in entries.items():
        for i in range(int(len(v[0]) / batch_size)):
            start_index = i*batch_size
            end_index = (i+1)*batch_size
            sub_data = v[0][start_index:end_index]
            sub_lens = v[1][start_index:end_index]
            tmp = [sub_data, sub_lens, k]
            data.append(tmp)

    for v in data:

        skuid = [v[2]]
        features = [v[0], skuid]

        yield  features, v[1]

def generate_epoch(data_path, num_epochs, batch_size):

    with open(data_path, 'rb') as f:
        entries = pickle.load(f)
    for epoch_num in range(num_epochs):
        yield generate_batch(batch_size, entries )

def generate_batch(batch_size , entries):
    """
    Generate batches of size <batch_size>.

    vocab_path = os.path.join(
        basedir, 'data/vocab.txt')
    train_data_path = os.path.join(
        basedir, 'data/train.p')
    validation_data_path = os.path.join(
        basedir, 'data/validation.p')
    vocab = Vocab(vocab_path)
    for k, v in entries.items():
        for v1 in v[0]:
            print("GENERATE", k, ''.join(vc.ids_to_tokens(v1, vocab=vocab)))
    """
    data = []
    for k, v in entries.items():
        for i in range(int(len(v[0]) / batch_size)):
            start_index = i*batch_size
            end_index = (i+1)*batch_size
            sub_data = v[0][start_index:end_index]
            sub_lens = v[1][start_index:end_index]
            tmp = [sub_data, sub_lens, k]
            data.append(tmp)

    random.shuffle(data)
    num_batches = len(data)

    for batch_num in range(num_batches):

        labels = [data[batch_num][2]]
        batch_features = [data[batch_num][0], labels]

        yield num_batches, batch_features, data[batch_num][1]

class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self):
        """
        """
        self.data_dir="data/"          # location of reviews data
        self.ckpt_dir="data/ckpt"      # location of model checkpoints
        self.mode="train"              # train|infer
        self.model="new"               # old|new
        self.lr=1e-4                   # learning rate
        #self.lr= 0.025                # learning rate
        self.num_epochs=200            # num of epochs 
        self.batch_size= 1000           # batch size
        self.hidden_size= 100          # num hidden units for RNN
        #self.embedding="glove"        # random|glove
        self.embedding="random"        # random|glove
        self.emb_size= 200             # num hidden units for embeddings
        self.max_grad_norm=5           # max gradient norm
        #self.keep_prob=0.9             # Keep prob for dropout layers
        self.keep_prob=0.9             # Keep prob for dropout layers
        self.num_layers=2              # number of layers for recurrsion
        self.max_input_length=40       # max number of words per review
        self.min_lr=1e-6               # minimum learning rate
        self.decay_rate=0.96           # Decay rate for lr per global step (train batch)
        self.save_every=10             # Save the model every <save_every> epochs
        self.model_name="imdb_model"   # Name of the model
        self.num_classes=5             # number of class for classify
        self.infer_dir = "data/ckpt"

def create_model(sess, FLAGS, vocab_size):
    """
    Creates a new model or loads old one.
    """
    imdb_model = Model(FLAGS, vocab_size)
    imdb_model._build_graph()
    logger.info("create_model build graph ok")

    if FLAGS.model == 'new':
        logger.info('==> Created a new model.')
        sess.run(tf.global_variables_initializer())
    elif FLAGS.model == 'old':
        ckpt = tf.train.get_checkpoint_state(
            os.path.join(basedir, FLAGS.ckpt_dir))
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("==> Restoring old model parameters from %s",
                ckpt.model_checkpoint_path)
            imdb_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("==> No old model to load from so initializing a new one.")
            sess.run(tf.global_variables_initializer())
    elif FLAGS.model == 'infer':
        ckpt = tf.train.get_checkpoint_state(
            os.path.join(basedir, FLAGS.infer_dir))
        if ckpt and ckpt.model_checkpoint_path:
            logger.info("==> Restoring infer model parameters from %s",
                ckpt.model_checkpoint_path)
            imdb_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("==> No infer model to load from so initializing a new one.")
            sess.run(tf.global_variables_initializer())

    return imdb_model

def train(FLAGS):
    """
    Train a previous or new model.
    """
    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/vocab.txt')
    train_data_path = os.path.join(
        basedir, 'data/train.p')
    validation_data_path = os.path.join(
        basedir, 'data/validation.p')
    vocab = Vocab(vocab_path)
    #FLAGS.num_classes = 5

    # Load embeddings (if using GloVe)
    if FLAGS.embedding == 'glove':
        with open(os.path.join(
            basedir, 'data/embeddings.p'), 'rb') as f:
            embeddings = pickle.load(f)
        FLAGS.vocab_size = len(embeddings)
    embeddings = np.zeros((len(vocab), FLAGS.emb_size)) 
    FLAGS.vocab_size = len(vocab)

    with tf.Session() as sess:

        # Create|reload model
        imdb_model = create_model(sess, FLAGS, len(vocab))

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join("log", time.strftime("%Y-%m-%d-%H-%M-%S")), sess.graph)
        #tf.initialize_all_variables().run()

        # Store attention score history for few samples
        #content = {"review": None, "label": None, "review_len": None, "attn_scores": None}
        attn_history_word = {"sample_%i"%i:{"review": None, "label": None, "review_len": None, "attn_scores": None} for i in range(FLAGS.batch_size)}

        #print(attn_history_word)
        # Start training
        for  train_epoch_num, train_epoch in \
            enumerate(generate_epoch(
                train_data_path, FLAGS.num_epochs, FLAGS.batch_size)):

            logger.info ("==> EPOCH: %s ", train_epoch_num )
            train_acc_count = 0
            train_batch_total = 0
            valid_acc_count = 0
            valid_batch_total = 0

            for  train_batch_num,  (total_batch, batch_features, batch_seq_lens) in \
                enumerate(train_epoch):
                #sys.exit()
                batch_reviews, batch_labels = batch_features
                batch_review_lens = batch_seq_lens

                # Display shapes once
                #for v in batch_reviews:
                #    print("TRAIN EPOCH:", train_epoch_num,"LABEL",  batch_labels, ''.join(vc.ids_to_tokens(v,vocab=vocab)))
                if (train_epoch_num == 0 and train_batch_num == 0):
                    logger.info ("Reviews: :%s", np.shape(batch_reviews))
                    logger.info ("Labels: %s", np.shape(batch_labels))
                    logger.info ("Review lens: %s", np.shape(batch_review_lens))

                _, train_logits, train_loss, train_acc,lr, attn_word_scores,attn_cmt_scores, logits,outputs,prob, distance = \
                imdb_model.train(
                        sess=sess,
                        batch_reviews=batch_reviews,
                        batch_labels=batch_labels,
                        batch_review_lens=batch_review_lens,
                        embeddings=embeddings,
                        keep_prob=FLAGS.keep_prob,
                        )
                logger.info("[TRAIN]: %i/%i|[ACC]: %.3f|[LOSS]: %.3f|[LABELS] : %i| %s|%s",
                       total_batch, train_batch_num,  train_acc, train_loss, batch_labels[0], distance,prob )

                train_batch_total += 1
                if train_acc > 0.99:
                    train_acc_count += 1
                if batch_labels[0] == 3:
                    for i in range(FLAGS.batch_size):
                        sample = "sample_%i"%i
                        attn_history_word[sample]["review"] = batch_reviews[i]
                        attn_history_word[sample]["label"] = batch_labels
                        attn_history_word[sample]["review_len"] = batch_review_lens[i]
                        attn_history_word[sample]["attn_scores"] = attn_word_scores[i]
                    attn_history_comment = attn_cmt_scores

            for valid_epoch_num, valid_epoch in \
                enumerate(generate_epoch(
                    data_path=validation_data_path,
                    num_epochs=1,
                    batch_size=FLAGS.batch_size,
                    )):

                for  valid_batch_num, (total_batch, valid_batch_features, valid_batch_seq_lens) in \
                    enumerate(valid_epoch):

                    valid_batch_reviews, valid_batch_labels = valid_batch_features
                    valid_batch_review_lens = valid_batch_seq_lens

                    #for v in valid_batch_reviews:
                    #    print("VALID EPOCH:", train_epoch_num,"LABEL",  valid_batch_labels, ''.join(vc.ids_to_tokens(v,vocab=vocab)))

                    valid_logits, valid_loss, valid_acc, prob = imdb_model.eval(
                        sess=sess,
                        batch_reviews=valid_batch_reviews,
                        batch_labels=valid_batch_labels,
                        batch_review_lens=valid_batch_review_lens,
                        embeddings=embeddings,
                        keep_prob=1.0, # no dropout for val|test
                        )
                    logger.info ("[VALID]: %i| [ACC]: %.3f | [LOSS]: %.6f,| [LABELS] : %i |%s", valid_batch_num , valid_acc,  valid_loss, valid_batch_labels[0], prob)
                    valid_batch_total += 1
                    if valid_acc > 0.99:
                        valid_acc_count += 1

            logger.info ("[EPOCH]: %i, [LR]: %.6e, [TRAIN ACC]: %.3f, [VALID ACC]: %.3f " \
                    "[TRAIN LOSS]: %.6f, [VALID LOSS]: %.6f " ,
                    train_epoch_num, lr, train_acc_count / train_batch_total, valid_acc_count/valid_batch_total, train_loss, valid_loss)

            # Save the model (maybe)
            if ((train_epoch_num == (FLAGS.num_epochs-1)) or
            ((train_epoch_num%FLAGS.save_every == 0) and (train_epoch_num>0))):

                # Make parents ckpt dir if it does not exist
                if not os.path.isdir(os.path.join(basedir, FLAGS.data_dir, 'ckpt')):
                    os.makedirs(os.path.join(basedir, FLAGS.data_dir, 'ckpt'))

                # Make child ckpt dir for this specific model
                if not os.path.isdir(os.path.join(basedir, FLAGS.ckpt_dir)):
                    os.makedirs(os.path.join(basedir, FLAGS.ckpt_dir))

                checkpoint_path = \
                    os.path.join(
                        basedir, FLAGS.ckpt_dir, "%s.ckpt" % FLAGS.model_name)

                logger.info ("==> Saving the model.")
                imdb_model.saver.save(sess, checkpoint_path,
                                 global_step=imdb_model.global_step)

                attn_word_file = os.path.join(basedir, FLAGS.ckpt_dir, 'attn_word_history.p')
                with open(attn_word_file, 'wb') as f:
                    pickle.dump(attn_history_word, f)

                attn_comment_file = os.path.join(basedir, FLAGS.ckpt_dir, 'attn_cmt_history.p')
                with open(attn_comment_file, 'wb') as f:
                    pickle.dump(attn_history_comment, f)


def infer(FLAGS):
    """
    Infer a previous or new model.
    """
    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/vocab.txt')
    validation_data_path = os.path.join(
        basedir, 'data/infer.p')
    vocab = Vocab(vocab_path)

    # Load embeddings (if using GloVe)
    embeddings = np.zeros((len(vocab), FLAGS.emb_size))
    FLAGS.vocab_size = len(vocab)

    with tf.Session() as sess:
        # Create|reload model
        imdb_model = create_model(sess, FLAGS, len(vocab))

        for  infer_index, infer_data in \
            enumerate(infer_data(
                infer_data_path, FLAGS.batch_size)):

            comments, skuid = valid_batch_features
            review_lens = valid_batch_seq_lens

            valid_logits, valid_loss, valid_acc, prob = imdb_model.infer(
                sess=sess,
                batch_reviews=valid_batch_reviews,
                batch_labels=valid_batch_labels,
                batch_review_lens=valid_batch_review_lens,
                embeddings=embeddings,
                keep_prob=1.0, # no dropout for val|test
                )
            logger.info ("[VALID]: %i| [ACC]: %.3f | [LOSS]: %.6f,| [LABELS] : %i |%s", valid_batch_num , valid_acc,  valid_loss, valid_batch_labels[0], prob)

if __name__ == '__main__':
    FLAGS = parameters()
    train(FLAGS)
