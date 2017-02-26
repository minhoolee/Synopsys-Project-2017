import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import argparse

def read_data_file(file_name):
    # Skip the header (column description) and column 0 (epoch column)
    return np.genfromtxt(file_name, delimiter=',', 
                         skip_header=1, usecols=(1, 2, 3, 4), 
                         names=['acc', 'loss', 'val_acc', 'val_loss'])

# TODO: make this accept a list of csv files and plot each one on the same axes
def plot_train_valid(csv_file='train.csv',
                     loss_png='loss.png',
                     acc_png='acc.png'):
    """
    Plot the training and validation losses and accuracies
    of a model from a CSV file
    """

    # Import the CSV containing the data
    data = read_data_file(csv_file)

    # Style with seaborn
    sns.set_style("darkgrid")

    # Summarize history for accuracy
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(acc_png, bbox_inches='tight')
    # plt.show()

    # Summarize history for loss
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(loss_png, bbox_inches='tight')
    # plt.show()

def plot_batch(csv_file='train.csv',
               loss_png='loss.png',
               acc_png='acc.png'):
    """
    Plot the training losses and accuracies of each batch from a CSV file
    """
    # Import the CSV containing the data
    data = read_data_file(csv_file)

    # Style with seaborn
    sns.set_style("darkgrid")

    # Summarize history for accuracy
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(acc_png, bbox_inches='tight')
    # plt.show()

    # Summarize history for loss
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(loss_png, bbox_inches='tight')
    # plt.show()
    


def main(argv):
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run a convolutional neural network on ' 
        + 'genetic sequences to derive epigenetic mechanisms')

    parser.add_argument('csv_file', metavar='CSV_FILE', help='The file (.csv) that stores the model\'s training and validation losses and accuracies')
    parser.add_argument('loss_png', metavar='LOSS_PNG', help='The file (.png) to store the model\'s training and validation accuracies graph')
    parser.add_argument('acc_png', metavar='ACC_PNG', help='The file (.png) to store the model\'s training and validation losses graph')

    args = parser.parse_args()

    plot_train_valid(csv_file=args.csv_file, loss_png=args.loss_png, acc_png=args.acc_png)

if __name__ == '__main__':
    main(sys.argv[1:])
