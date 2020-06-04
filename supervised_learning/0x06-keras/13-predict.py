#!/usr/bin/env python3
"""Makes a prediction using Keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """network is the network being used to make the prediction,
    data is the data to predict, verbose is whether or not to
    print the progress"""
    return network.predict(x=data, verbose=verbose)
