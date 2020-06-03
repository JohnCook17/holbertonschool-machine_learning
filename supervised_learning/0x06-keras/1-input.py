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
    for i in range(0, len(layers)):
        x = K.layers.Dense(units=layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=(1 - keep_prob), noise_shape=(layers[i],)
                                 )(x)
    outputs = x
    return K.Model(inputs=inputs, outputs=outputs)
