"""Train model for sentiment analysis.

Usage:
  train.py [--epochs=N] [--train-embedding] [--glove=FILE] [--data-cache=FILE]
           [--model=NAME] [--force-preprocess] [--parameters=FILE]
           [--evaluate-only] [--continue-training] [--hidden-dims=N]
  train.py (-h | --help)

Options:
  --epochs=N            Number of epochs [default: 10]
  --glove=FILE          Path to file with glove embedding to use. [default: glove.6B/glove.6B.50d.txt]
  --model=NAME          Name of the model to use (should have a corresponding .py file in models package). [default: lstm]
  --data-cache=FILE     Path to cached processed data. If file doesn't exist, reprocess data and save to this location. [default: sst_data.pkl]
  --parameters=FILE     Path to saved parameters. Neither --continue-training nor --evaluate-only are provided, the parameters will be overwritten. [default: lstm.tflearn]
  --hidden-dims=N        Number of dimensions to use in the hidden state. [default: 128]
  --continue-training   Continue training with the saved parameters.
  --evaluate-only       Skip training and only evaluate the saved model.
  --train-embedding     Train the word embedding along with model.
  --force-preprocess    Delete the cached data if it exists and preprocess the data again.
  -h --help             Show this screen.
"""
from __future__ import division, print_function
import cPickle as pkl
import os

from docopt import docopt
from sklearn.metrics import classification_report
import tflearn

import utils
import models


def load_data(args):
    print("Loading glove vectors from %r..." % args['--glove'])
    glove = utils.build_glove(args['--glove'])

    if not args['--force-preprocess'] and os.path.exists(args['--data-cache']):
        print("Loading preprocessed data...")
        with open(args['--data-cache']) as f:
            data = pkl.load(f)
    else:
        print("Processing treebank dataset...")
        data = utils.load_sst(glove)
        print("Caching preprocessed data...")
        with open(args['--data-cache'], 'w') as f:
            pkl.dump(data, f)
    return data, glove


def train(args, glove, data):
    print("Loading model definition for %s..." % args['--model'])
    net = models.get_model(args['--model'], embedding_size=glove[0].shape,
                           train_embedding=args['--train-embedding'],
                           hidden_dims=int(args['--hidden-dims']))
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

    if args['--evaluate-only'] or args['--continue-training']:
        print("Loading saved parameters..")
        model.load(args['--parameters'])
    else:
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

        print("Saving parameters to %r..." % args['--parameters'])
        model.save(args['--parameters'])

    return model


def evaluate(args, model, data):
    train_predict = model.predict(data.trainX)
    print("TRAINING RESULTS")
    print(classification_report(
        [utils.get_sentiment(e[1]) for e in train_predict],
        [e[1] for e in data.trainY]))
    print()

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

    data, glove = load_data(args)
    model = train(args, glove, data)
    evaluate(args, model, data)


if __name__ == '__main__':
    main()
