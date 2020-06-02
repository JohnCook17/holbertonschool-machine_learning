#!/usr/bin/env python3
"""Builds a model in keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a model with nx being the input shape. layers is the shape of
    the layers. activations being the activation to use per layer.
    lambtha is the l2 regularization. keep_prob is the probability to keep
    a neuron."""
    model = K.Sequential()
    first_layer = 1
    for layer, activation in zip(layers, activations):
        if first_layer:
            model.add(K.layers.Dense(units=layer, activation=activation,
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha),
                                     input_shape=(nx,)))
            first_layer = 0
        else:
            model.add(K.layers.Dense(units=layer, activation=activation,
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha)))
        if activation != "softmax":
            model.add(K.layers.Dropout(rate=(1 - keep_prob),
                                       noise_shape=(layer,), input_shape=(nx,)
                                       ))
    return model