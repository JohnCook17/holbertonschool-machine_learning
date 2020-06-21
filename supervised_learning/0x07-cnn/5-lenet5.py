#!/usr/bin/env python3
"""lenet5 in keras"""
import tensorflow.keras as K


def lenet5(X):
    """Keras version of lenet5"""
    initializer = K.initializers.he_normal()
    inputs = X
    layer_0 = K.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              activation="relu",
                              padding="SAME",
                              kernel_initializer=initializer)(inputs)
    layer_1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer_0)
    layer_2 = K.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              activation="relu",
                              padding="VALID",
                              kernel_initializer=initializer)(layer_1)
    layer_3 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2))(layer_2)
    layer_4 = K.layers.Flatten()(layer_3)
    layer_5 = K.layers.Dense(units=120,
                             activation="relu",
                             kernel_initializer=initializer)(layer_4)
    layer_6 = K.layers.Dense(units=84,
                             activation="relu",
                             kernel_initializer=initializer)(layer_5)
    outputs = K.layers.Dense(units=10,
                             activation="softmax",
                             kernel_initializer=initializer)(layer_6)
    model = K.Model(inputs=inputs, outputs=outputs)
    opt = K.optimizers.Adam()
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
