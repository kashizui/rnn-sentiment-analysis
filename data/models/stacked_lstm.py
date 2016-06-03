# -*- coding: utf-8 -*-
"""
Two-layer stacked LSTM.
"""
from __future__ import division, print_function
import tflearn


def build(embedding_size=(400000, 50), train_embedding=False, hidden_dims=128,
          learning_rate=0.001):
    net = tflearn.input_data([None, 200])
    net = tflearn.embedding(net, input_dim=embedding_size[0],
                            output_dim=embedding_size[1],
                            trainable=train_embedding, name='EmbeddingLayer')
    net = tflearn.lstm(net, hidden_dims, return_seq=True)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.lstm(net, hidden_dims)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy')
    return net


