"""Train model for sentiment analysis.

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


def train(d, model_name):
    net = models.get_model(model_name)

    # Training
    print("Training...")
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
    model.fit(d.trainX, d.trainY, validation_set=(d.valX, d.valY),
              show_metric=True, batch_size=128)

    print("Saving Model...")
    model.save('lstm.tflearn')


def main():
    args = docopt(__doc__)
    dataset = utils.load_sst('sst_data.pkl')
    train(dataset, args['<model>'])


if __name__ == '__main__':
    main()
