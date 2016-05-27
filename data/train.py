"""Train model for sentiment analysis.

Try `python train.py lstm` first.

Usage:
  train.py <model>
  train.py (-h | --help)

Options:
  -h --help     Show this screen.
"""
from __future__ import division, print_function, absolute_import

from docopt import docopt
import tflearn

import utils
import models


def main():
    args = docopt(__doc__)
    data = utils.load_sst('sst_data.pkl')

    print("Loading model definition for %s..." % args['<model>'])
    net = models.get_model(args['<model>'])
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

    print("Training...")
    model.fit(data.trainX, data.trainY,
              validation_set=(data.valX, data.valY),
              show_metric=True, batch_size=128)

    print("Saving Model...")
    model_path = '%s.tflearn' % args['<model>']
    model.save(model_path)
    print("Saved model to %s" % model_path)


if __name__ == '__main__':
    main()
