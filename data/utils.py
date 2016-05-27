"""
Utility functions.
"""
from __future__ import division, print_function
from collections import namedtuple
import cPickle as pkl
import csv

import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
import random


Dataset = namedtuple('Dataset', 'trainX, trainY, valX, valY, testX, testY')


def get_sentiment(val):
    return int(val >= 0.5)


def format_data(dataset):
    x = [item[0] for item in dataset]
    y = [item[1] for item in dataset]
    return x, y


def load_sst(fname):
    """Loads processed Stanford Sentiment Treebank data."""
    print("Loading Data...")
    data = pkl.load(open(fname, "rb"))
    train = data['train']
    test = data['test']
    val = data['dev']

    trainX, trainY = format_data(train[1:])
    valX, valY = format_data(val[1:])
    testX, testY = format_data(test[1:])

    # Data preprocessing
    # Sequence padding
    print("Padding Sequences...")
    trainX = pad_sequences(trainX, maxlen=200, value=0.)
    valX = pad_sequences(valX, maxlen=200, value=0.)
    testX = pad_sequences(testX, maxlen=200, value=0.)

    # Converting labels to binary vectors
    print("Converting labels to binary vectors...")
    trainY = to_categorical(trainY, nb_classes=2)
    valY = to_categorical(valY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    return Dataset(trainX, trainY, valX, valY, testX, testY)


def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
    """Reads in matrices from CSV or space-delimited files.

    Parameters
    ----------
    src_filename : str
        Full path to the file to read.

    delimiter : str (default: ',')
        Delimiter for fields in src_filename. Use delimter=' '
        for GloVe files.

    header : bool (default: True)
        Whether the file's first row contains column names.
        Use header=False for GloVe files.

    quoting : csv style (default: QUOTE_MINIMAL)
        Use the default for normal csv files and csv.QUOTE_NONE for
        GloVe files.

    Returns
    -------
    (np.array, list of str, list of str)
       The first member is a dense 2d Numpy array, and the second
       and third are lists of strings (row names and column names,
       respectively). The third (column names) is None if the
       input file has no header. The row names are assumed always
       to be present in the leftmost column.
    """
    reader = csv.reader(open(src_filename), delimiter=delimiter,
                        quoting=quoting)
    colnames = None
    if header:
        colnames = next(reader)
        colnames = colnames[1:]
    mat = []
    rownames = []
    for line in reader:
        rownames.append(line[0])
        mat.append(np.array(list(map(float, line[1:]))))
    return np.array(mat), rownames, colnames


def build_glove(src_filename):
    """Wrapper for using `build` to read in a GloVe file as a matrix"""
    return build(src_filename, delimiter=' ', header=False,
                 quoting=csv.QUOTE_NONE)


def glove2dict(src_filename):
    """GloVe Reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors.

    """
    reader = csv.reader(open(src_filename), delimiter=' ',
                        quoting=csv.QUOTE_NONE)
    return {line[0]: np.array(list(map(float, line[1:]))) for line in reader}


def randvec(n=50, lower=-0.5, upper=0.5):
    """Returns a random vector of length `n`. `w` is ignored."""
    return np.array([random.uniform(lower, upper) for i in range(n)])