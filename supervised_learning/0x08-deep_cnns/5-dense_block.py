#!/usr/bin/env python3
"""A denselayer dense block in tensorflow keras"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """A dense block, X is the previous layer, nb_filters is the number of
    filters to use, growth rate is the rate to change the number of filters
    by, and layers is how many layers"""
    initializer = K.initializers.he_normal()
    filter_total = 0
    for layer in range(1, layers + 1):
        batch_normalization = K.layers.BatchNormalization()(X)
        activation = K.layers.Activation("relu")(batch_normalization)
        conv2d = K.layers.Conv2D(filters=nb_filters + growth_rate * (layer - 1),
                                 kernel_size=(1, 1),
                                 padding="same",
                                 activation="relu",
                                 kernel_initializer=initializer)(activation)
        batch_normalization1 = K.layers.BatchNormalization()(conv2d)
        activation1 = K.layers.Activation("relu")(batch_normalization1)
        conv2d1 = K.layers.Conv2D(filters=growth_rate,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu",
                                  kernel_initializer=initializer)(activation1)
        concatenate = K.layers.concatenate([X, conv2d1])
        X = concatenate
        filter_total = growth_rate + (nb_filters + growth_rate * (layer - 1))
    return X, filter_total
