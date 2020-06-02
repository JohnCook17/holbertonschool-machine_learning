#!/usr/bin/env python3
""""""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """"""
    model = K.Sequential()
    for layer, activation in zip(layers, activations):
        model.add(K.layers.Dense(units=layer, activation=activation,
                                 kernel_regularizer=K.regularizers.l2(lambtha)))
        model.add(K.layers.Dropout(rate=keep_prob, noise_shape=layer))
    model.build((None, nx))
    return model
