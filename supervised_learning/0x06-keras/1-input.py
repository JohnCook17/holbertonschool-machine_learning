#!/usr/bin/env python3
""""""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """"""
    inputs = K.Input(shape=(nx,))
    x = inputs
    for layer, activation in zip(layers, activations):
        x = K.layers.Dense(units=layer, activation=activation, kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if activation != "softmax":
            x = K.layers.Dropout(rate=(1 - keep_prob),noise_shape=(layer,))(x)
    outputs = x
    return K.Model(inputs=inputs, outputs=outputs)
