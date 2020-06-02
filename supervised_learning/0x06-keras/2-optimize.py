#!/usr/bin/env python3
"""Uses keras.Optimizer to make an adam optimizer"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """network is the model to optimize
    alpha is the learning rate
    beta1 is the first adam optimization parameter
    beta2 is the second adam optimization parameter"""
    return K.Optimizer.Adam(alpha, beta1, beta2)(network)
