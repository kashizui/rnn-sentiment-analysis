"""Train model for sentiment analysis.

Usage:
  train.py [--epochs=N] [--train-embedding] [--glove=FILE]
           [--model=NAME] [--force-preprocess]
           [--parameters=FILE] [--evaluate-only] [--continue-training]
           [--hidden-dims=N] [--evaluate-test] [--embedding-dims=N]
  train.py (-h | --help)

Options:
  --epochs=N            Number of epochs [default: 10]
  --glove=FILE          Path to file with glove embedding to use. [default: glove.6B/glove.6B.50d.txt]
  --model=NAME          Name of the model to use (should have a corresponding .py file in models package). [default: lstm]
  --parameters=FILE     Path to saved parameters. Neither --continue-training nor --evaluate-only are provided, the parameters will be overwritten.
  --hidden-dims=N       Number of dimensions to use in the hidden state. [default: 128]
  --embedding-dims=N    Number of dimensions in randomly initialized word embedding (using this option supersedes glove embedding).
  --continue-training   Continue training with the saved parameters.
  --evaluate-only       Skip training and only evaluate the saved model.
  --train-embedding     Train the word embedding along with model.
  --force-preprocess    Delete the cached data if it exists and preprocess the data again.
  --evaluate-test       Evaluate the test set (only do this for final evaluations!)
  -h --help             Show this screen.
"""
from __future__ import division, print_function
import cPickle as pkl
import hashlib
import json
import os
import time

from docopt import docopt
from sklearn.metrics import classification_report
import tflearn

import utils
import models


# The only params that actually affect the data loading and results
REAL_PARAMS = (
    '--glove',
    '--model',
    '--hidden-dims',
    '--embedding-dims',
    '--train-embedding',
)

CACHE_DIR = '/tmp/rnncache'


def get_arg_hash(args):
    args = {k: args[k] for k in REAL_PARAMS}
    return hashlib.md5(json.dumps(args)).hexdigest()


def load_data(args):
    if args['--embedding-dims'] is None:
        print("Loading glove vectors from %r..." % args['--glove'])
        glove = utils.build_glove(args['--glove'])
    else:
        # We will randomly initialize word embedding, no need for glove.
        print("Using random word embedding.")
        glove = None

    # Create CACHE_DIR if not exists
    if not os.path.isdir(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    # Use hash of arguments as filename of data cache
    cache_file_path = os.path.join(CACHE_DIR, get_arg_hash(args) + '.pkl')
    print("Data cache file: " + cache_file_path)

    if not args['--force-preprocess'] and os.path.exists(cache_file_path):
        print("Loading preprocessed data...")
        with open(cache_file_path) as f:
            data = pkl.load(f)
    else:
        print("Processing treebank dataset...")
        data = utils.load_sst(glove)
        print("Caching preprocessed data...")
        with open(cache_file_path, 'w') as f:
            pkl.dump(data, f)
    return data, glove


def train(args, glove, data, param_file_path):
    if glove is None:
        embedding_size = (utils.NUM_UNIQUE_TOKENS, int(args['--embedding-dims']))
    else:
        embedding_size = glove[0].shape

    print("Loading model definition for %s..." % args['--model'])
    net = models.get_model(args['--model'], embedding_size=embedding_size,
                           train_embedding=args['--train-embedding'],
                           hidden_dims=int(args['--hidden-dims']))
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

    if args['--evaluate-only'] or args['--continue-training']:
        print("Loading saved parameters from %s" % param_file_path)
        model.load(param_file_path)
    elif glove is not None:
        print("Initializing word embedding...")
        # Retrieve embedding layer weights (only a single weight matrix, so index is 0)
        embedding_weights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
        # Initialize with glove embedding
        model.set_weights(embedding_weights, glove[0])

    if not args['--evaluate-only']:
        print("Training...")
        model.fit(data.trainX, data.trainY,
                  n_epoch=int(args['--epochs']),
                  validation_set=(data.valX, data.valY),
                  show_metric=True, batch_size=128)

        print("Saving parameters to %s" % param_file_path)
        model.save(param_file_path)

    return model


def evaluate(args, model, data):
    train_predict = model.predict(data.trainX)
    print("TRAINING RESULTS")
    print(classification_report(
        [utils.get_sentiment(e[1]) for e in train_predict],
        [e[1] for e in data.trainY]))
    print()

    test_predict = model.predict(data.valX)
    print("DEV RESULTS")
    print(classification_report(
        [utils.get_sentiment(e[1]) for e in test_predict],
        [e[1] for e in data.valY]))
    print()

    if args['--evaluate-test']:
        test_predict = model.predict(data.testX)
        print("TEST RESULTS")
        print(classification_report(
            [utils.get_sentiment(e[1]) for e in test_predict],
            [e[1] for e in data.testY]))
        print()


def main():
    args = docopt(__doc__)
    # Print argument list for debugging
    for k, v in args.iteritems():
        print("{:<20}: {!r:<10}".format(k, v))

    if args['--parameters'] is None:
        param_file_path = 'run%d.tflearn' % int(time.time())
    else:
        param_file_path = args['--parameters']
    log_file_path = param_file_path + '.log'
    utils.Tee(log_file_path, 'a')
    print("Parameters file: " + param_file_path)
    print("Log file: " + log_file_path)

    data, glove = load_data(args)
    model = train(args, glove, data, param_file_path)
    evaluate(args, model, data)


if __name__ == '__main__':
    main()
