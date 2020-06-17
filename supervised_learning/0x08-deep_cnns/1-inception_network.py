#!/usr/bin/env python3
"""Inception network in tensorflow keras"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """The inception network"""
    initializer = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    conv0 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding="same",
                            activation="relu",
                            kernel_initializer=initializer)(inputs)
    maxpool0 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding="same")(conv0)
    conv1a = K.layers.Conv2D(filters=64,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding="same",
                             activation="relu",
                             kernel_initializer=initializer)(maxpool0)
    conv1b = K.layers.Conv2D(filters=192,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding="same",
                             activation="relu",
                             kernel_initializer=initializer)(conv1a)
    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding="same")(conv1b)
    inception_3a = inception_block(maxpool1, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])
    maxpool2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding="same")(inception_3b)
    inception_4a = inception_block(maxpool2, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    maxpool3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding="same")(inception_4e)
    inception_5a = inception_block(maxpool3, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1))(inception_5b)
    drop40per = K.layers.Dropout(rate=0.4)(avgpool)
    softmax = K.layers.Dense(units=1000,
                             activation="softmax",
                             kernel_initializer=initializer)(drop40per)
    model = K.Model(inputs=inputs, outputs=softmax)
    return model
