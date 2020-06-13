#!/usr/bin/env python3
"""lenet5 in keras"""
import tensorflow.keras as K


def lenet5(X):
    """Keras version of lenet5"""
    model = K.Sequential()
    model.inputs = X
    model.add(K.layers.Conv2D(filters=6, kernel_size=(5, 5), activation="relu",
                              padding="SAME"))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                              activation="relu", padding="VALID"))
    model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(units=120, activation="relu"))
    model.add(K.layers.Dense(units=84, activation="relu"))
    model.add(K.layers.Dense(units=10, activation="softmax"))
    opt = K.optimizers.Adam()
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
