import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
basedir = ""
class parameters():
    """
    Arguments for data processing.
    """
    def __init__(self):
        self.ckpt_dir="../data/ckpt"
        # location of model checkpoints
        self.model_name="imdb_model"
        # Name of the model

def plot_metrics(FLAGS):
    """
    Plot the loss and accuracy for train|test.
    """
    import seaborn as sns

    # Load metrics from file
    metrics_file = os.path.join(basedir, FLAGS.ckpt_dir, 'metrics.p')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    # Plot results
    ax1 = axes[0]
    ax1.plot(metrics["train_acc"], label='train accuracy')
    ax1.plot(metrics["valid_acc"], label='valid accuracy')
    ax1.legend(loc=4)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('train|valid accuracy')
    ax2 = axes[1]
    ax2.plot(metrics["train_loss"], label='train loss')
    ax2.plot(metrics["valid_loss"], label='valid loss')
    ax2.legend(loc=3)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('train|valid loss')
    plt.show()
    plt.savefig("image/metrics.png", dpi=150)

FLAGS = parameters()
# Add model name to ckpt dir
#FLAGS.ckpt_dir = FLAGS.ckpt_dir + '/%s'%(FLAGS.model_name)
plot_metrics(FLAGS)
