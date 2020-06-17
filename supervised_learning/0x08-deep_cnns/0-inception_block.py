#!/usr/bin/env python3
"""An inception block in tf keras"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """A_prev is the previous layer and filters is the number of filters to use
    """
    initializer = K.initializers.he_normal()
    conv2d_1 = K.layers.Conv2D(filters=filters[1],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same",
                               activation="relu",
                               kernel_initializer=initializer)(A_prev)
    conv2d_3 = K.layers.Conv2D(filters=filters[3],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same",
                               activation="relu",
                               kernel_initializer=initializer)(A_prev)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(1, 1),
                                          padding="same")(A_prev)
    conv2d = K.layers.Conv2D(filters=filters[0],
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding="same",
                             activation="relu",
                             kernel_initializer=initializer)(A_prev)
    conv2d_2 = K.layers.Conv2D(filters=filters[2],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same",
                               activation="relu",
                               kernel_initializer=initializer)(conv2d_1)
    conv2d_4 = K.layers.Conv2D(filters=filters[4],
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding="same",
                               activation="relu",
                               kernel_initializer=initializer)(conv2d_3)
    conv2d_5 = K.layers.Conv2D(filters=filters[5],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same",
                               activation="relu",
                               kernel_initializer=initializer)(max_pooling2d)
    concatenate = K.layers.concatenate([conv2d, conv2d_2, conv2d_4, conv2d_5])
    return concatenate
