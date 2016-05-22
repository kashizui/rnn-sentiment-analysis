# -*- coding: utf-8 -*-
"""Evaluate model for sentiment analysis.

Usage:
  evaluate.py <model>
  evaluate.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
from __future__ import division, print_function, absolute_import
import cPickle as pkl

from docopt import docopt
import tflearn
from sklearn.metrics import classification_report

import utils
import models


def main():
    args = docopt(__doc__)
    data = utils.load_sst('sst_data.pkl')
    model_name = args['<model>'].split('.')[0]
    net = models.get_model(model_name)

    print("Loading model definition for %s..." % model_name)
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

    model_path = '%s.tflearn' % model_name
    print("Loading saved model at %r..." % model_path)
    model.load(model_path)

    print("Evaluating model...")
    test_predict = model.predict(data.testX)

    y_pred = [e[0] > 0.5 for e in test_predict]
    y_true = [e[0] for e in data.testY]

    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    main()


