#!/usr/bin/env python3
"""identity block in tensorflow keras"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """A_prev is the previous layer, filters are the filters to use"""
    initializer = K.initializers.he_normal()
    conv2d = K.layers.Conv2D(filters=filters[0],
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding="same",
                             kernel_initializer=initializer)(A_prev)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation("relu")(batch_normalization)
    conv2d_1 = K.layers.Conv2D(filters=filters[1],
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=initializer)(activation)
    batch_normalization_1 = K.layers.BatchNormalization()(conv2d_1)
    activation_1 = K.layers.Activation("relu")(batch_normalization_1)
    conv2d_2 = K.layers.Conv2D(filters=filters[2],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same",
                               kernel_initializer=initializer)(activation_1)
    batch_normalization_2 = K.layers.BatchNormalization()(conv2d_2)
    add = K.layers.Add()([batch_normalization_2, A_prev])
    activation_2 = K.layers.Activation("relu")(add)
    return activation_2