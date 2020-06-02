#!/usr/bin/env python3
"""Uses keras.Model to make a model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """nx is the size of the input, layers is a list of the
    size of the layres, activations is a list of which activation
    to use, lambtha is the l2 regularization, keep prob is the
    dropout rate"""
    inputs = K.Input(shape=(nx,))
    x = inputs
    for layer, activation in zip(layers, activations):
        x = K.layers.Dense(units=layer, activation=activation,
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if activation != "softmax":
            x = K.layers.Dropout(rate=(1 - keep_prob), noise_shape=(layer,))(x)
    outputs = x
    return K.Model(inputs=inputs, outputs=outputs)
