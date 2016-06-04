"""
Utility functions.
"""
from __future__ import division, print_function
from collections import namedtuple, defaultdict
import cPickle as pkl
import csv

import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
import random


Dataset = namedtuple('Dataset', 'trainX, trainY, valX, valY, testX, testY')

NUM_UNIQUE_TOKENS = 21701 + 1


def get_sentiment(val):
    return int(val >= 0.5)


def unzip_examples(dataset):
    x = [item[0] for item in dataset]
    y = [item[1] for item in dataset]
    return x, y


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
    mat, rownames, colnames = build(src_filename, delimiter=' ', header=False,
                                    quoting=csv.QUOTE_NONE)
    # Prepend row at index 0 for unknown tokens
    rownames.insert(0, '<unk>')
    mat = np.vstack([
        np.random.uniform(low=-0.5, high=0.5, size=mat[0].shape),
        mat
    ])
    return mat, rownames, colnames


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

NUM_EXAMPLES = float('inf')


def split_database(list_to_split, split_fname):
    """
    Split a list of sentences into train, test, and dev groups.

    Based on the index specified in split_fname, the sentences will
    be grouped according to index into lists.

    Arguments:
    ----------
        list_to_split: the list to be split into groups.
        split_fname: name of file with one line for each list element

    Returns:
    ----------
        train: Training set as divided by split_fname
        test: Test set as divided by split_fname
        dev: Dev set as divided by split_fname
    """
    train = []
    test = []
    dev = []
    with open(split_fname, 'rb') as split:
        next(split)
        for line in split:
            index, group = line.split(',')
            group = int(group.strip())
            if index in list_to_split:
                if group == 1:
                    train.append(list_to_split[index])
                elif group == 2:
                    test.append(list_to_split[index])
                elif group == 3:
                    dev.append(list_to_split[index])
    return train, test, dev


def phrases2ints(word_dict, dataset_sentences_fname):
    """
    Convert a dataset of sentences into numbers corresponding to the words in the sentence.

    Arguments:
    ----------
        word_dict: Dict relating words to their indices
        dataset_sentences_fname: File name that holds the sentences in the dataset.

    Returns:
    ----------
        int_phrases: Returns phrases with ints substituted for words corresponding to their index
                        in dataset_sentences_fname

    """
    with open(dataset_sentences_fname, 'rb') as dataset_sentences:
        int_phrases = []
        next(dataset_sentences)
        num_lines_read = 0
        for sentence in dataset_sentences:
            if num_lines_read >= NUM_EXAMPLES:
                break

            index, phrase = sentence.split('\t')
            int_phrase = tuple(
                word_dict.get(word.lower(), 0)  # assumes vector for unk token at row 0
                for word in phrase.split()
            )
            int_phrases.append((int_phrase, index))
            num_lines_read += 1

    return int_phrases


def get_phrases_dict(phrases_fname):
    """
    Convert a file with phrases into a phrase dict and a word dict.

    Arguments:
    ----------
        phrases_fname: Name of file that holds phrases and their corresponding index.

    Returns:
    ----------
        phrase_dict: Dict of phrases and their corresponding index
        word_dict: subset of phrase_dict that only contains single words
    """
    phrase_dict = {}
    with open(phrases_fname, 'rb') as phrases:
        for line in phrases.readlines():
            phrase, index = line.split('|')
            phrase_dict[phrase] = int(index.strip())
    word_dict = {key: val for key, val in phrase_dict.items() if len(key.split())==1}
    return phrase_dict, word_dict


def indices_to_sentiment(int_phrases, sentiment_labels_fname):
    with open(sentiment_labels_fname, 'rb') as sentiment_labels:
        next(sentiment_labels)
        label_dict = dict(tuple(item.split('|')) for item in sentiment_labels)

    return {
        idx: (phrase, get_sentiment(float(label_dict[idx])))  # FIXME
        for phrase, idx in int_phrases
        if idx.isdigit()
    }


def glove_word_indices(glove_data):
    words = glove_data[1]  # list of rownames
    return {word: i for i, word in enumerate(words)}


def increment_word_dict():
    class nonlocal:
        counter = -1

    def nexti():
        nonlocal.counter += 1
        return nonlocal.counter

    return defaultdict(nexti)


def load_sst(glove_data):
    # Get the phrases and their indices
    # print("Getting Phrase Dictionary...")
    # _, word_dict = get_phrases_dict('stanfordSentimentTreebank/dictionary.txt')

    if glove_data is None:
        word_dict = increment_word_dict()
    else:
        print("Getting glove word indices...")
        word_dict = glove_word_indices(glove_data)

    # Convert the phrases to ints with word indices so they can be processed by Neural Network
    print("Converting to Ints...")
    int_phrases = phrases2ints(word_dict, 'stanfordSentimentTreebank/datasetSentences.txt')

    # Convert indices to sentiment values
    print("Converting Indices to Sentiment Values...")
    phrase_sentiments = indices_to_sentiment(int_phrases, 'stanfordSentimentTreebank/sentiment_labels.txt')

    # Split into train, test, and dev groups
    print("Splitting into train, test, and dev groups...")
    train, test, val = split_database(phrase_sentiments, 'stanfordSentimentTreebank/datasetSplit.txt')

    # Unzip input sequences and sentiment labels
    trainX, trainY = unzip_examples(train)
    valX, valY = unzip_examples(val)
    testX, testY = unzip_examples(test)

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


# Hacky class to help duplicate stdout to log
import sys
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
