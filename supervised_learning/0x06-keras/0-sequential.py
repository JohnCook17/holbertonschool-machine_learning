#!/usr/bin/env python3
"""Builds a model in keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a model with nx being the input shape. layers is the shape of
    the layers. activations being the activation to use per layer.
    lambtha is the l2 regularization. keep_prob is the probability to keep
    a neuron."""
    model = K.Sequential()
    for i in range(0, len(layers)):
        if i == 0:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha),
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha)))
        if i == len(layers) - 1:
            break
        else:
            model.add(K.layers.Dropout(rate=(1 - keep_prob),
                                       noise_shape=(layers[i],),
                                       input_shape=(nx,)
                                       ))
    return model
