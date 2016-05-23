from __future__ import division, print_function, absolute_import
import gensim
from docopt import docopt
import tflearn
import utils
import models

 
def generate_embedding():
    with open('stanfordSentimentTreebank/datasetSentences.txt') as f:
        sentences = f.read().splitlines()
    embedding = gensim.models.Word2Vec(sentences)
    embedding = gensim.models.Word2Vec() # an empty model, no training
    return embedding


def generate_net(embedding):
    net = tflearn.input_data([None, 200])
    net = tflearn.embedding(net, input_dim=300000, output_dim=128)
    net = tflearn.lstm(net, 128)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy')
    return net


def train():
    embedding = generate_embedding()
    data = utils.load_sst('sst_data.pkl')
    net = generate_net(embedding)
    print("Loading model definition for %s..." % model)
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
    net = models.get_model(model)

    print("Training...")
    model.fit(data.trainX, data.trainY,
              validation_set=(data.valX, data.valY),
              show_metric=True, batch_size=128)

    print("Saving Model...")
    model_path = '%s.tflearn' % model
    model.save(model_path)
    print("Saved model to %s" % model_path)


def main():
	train()


if __name__ == '__main__':
    main()
