"""
Utility functions.
"""
from collections import namedtuple
import cPickle as pkl

from tflearn.data_utils import to_categorical, pad_sequences


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

