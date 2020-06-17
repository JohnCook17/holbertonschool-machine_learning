#!/usr/bin/env python3
""""""
import tensorflow.keras as K  # change this line!!!


def inception_block(A_prev, filters):
    """"""
    conv2d_1 = K.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(A_prev)
    conv2d_3 = K.layers.Conv2D(filters=filters[3], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(A_prev)
    max_pooling2d = K.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding="same")(A_prev)
    conv2d = K.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(A_prev)
    conv2d_2 = K.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(conv2d_1)
    conv2d_4 = K.layers.Conv2D(filters=filters[4], kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu")(conv2d_3)
    conv2d_5 = K.layers.Conv2D(filters=filters[5], kernel_size=(1, 1), strides=(1, 1), padding="same", activation="relu")(max_pooling2d)
    concatenate = K.layers.concatenate([conv2d, conv2d_2, conv2d_4, conv2d_5], axis=-1)
    return concatenate
