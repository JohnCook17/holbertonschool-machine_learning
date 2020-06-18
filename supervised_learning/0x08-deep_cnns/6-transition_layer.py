#!/usr/bin/env python3
"""A transition layer for densenet in keras"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """A transition layer, X is the previous layer, nb_filters is the number
    of filters, compression is how much to compress the layers by."""
    initializer = K.initializers.he_normal()
    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation("relu")(batch_normalization)
    conv2d = K.layers.Conv2D(filters=int(nb_filters * compression),
                             kernel_size=(1, 1),
                             padding="same",
                             kernel_initializer=initializer)(activation)
    average_pooling2d = K.layers.AveragePooling2D(pool_size=(2, 2),
                                                  strides=(2, 2),
                                                  padding="same")(conv2d)
    return average_pooling2d, int(nb_filters * compression)
