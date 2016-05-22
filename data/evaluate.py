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

import utils
import models


def main():
    args = docopt(__doc__)
    data = utils.load_sst('sst_data.pkl')
    net = models.get_model(args['<model>'])

    print("Loading model definition for %s..." % args['<model>'])
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

    print("Loading saved model...")
    model.load('%s.tflearn' % args['<model>'])

    # TODO(sckoo): evaluate
    print(model)

if __name__ == '__main__':
    main()


