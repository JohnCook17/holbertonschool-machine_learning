#!/usr/bin/env python3
"""Uses .fit to train the network"""


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """network is the network to train, data is the data, labels is the data
    labels, batch size is the size of the minibatch, epochs is the number of
    epochs to run, verbose is print of not the progress, shuffle is to shuffle
    the data or not"""
    return network.fit(x=data, y=labels, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, shuffle=shuffle)
