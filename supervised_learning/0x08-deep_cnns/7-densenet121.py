#!/usr/bin/env python3
""""""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """"""
    initializer = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    batch_normalization = K.layers.BatchNormalization()(inputs)
    activation = K.layers.Activation("relu")(batch_normalization)
    conv2d = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding="same",
                             kernel_initializer=initializer)(activation)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding="same")(conv2d)
    # Dense Block 1
    db_1, nb_filters = dense_block(max_pooling2d, 64, 32, 6)
    # Transition Layer 1
    tr_1, nb_filters = transition_layer(db_1, nb_filters, compression)
    # Dense Block 2
    db_2, nb_filters = dense_block(tr_1, nb_filters, 32, 12)
    # Transition Layer 2
    tr_2, nb_filters = transition_layer(db_2, nb_filters, compression)
    # Dense Block 3
    db_3, nb_filters = dense_block(tr_2, nb_filters, 32, 24)
    # Transition Layer 3
    tr_3, nb_filters = transition_layer(db_3, nb_filters, compression)
    # Dense Block 4
    db_4, nb_filters = dense_block(tr_3, nb_filters, 32, 16)
    # Classification Layer
    average_pooling2d = K.layers.AveragePooling2D(pool_size=(7, 7),
                                                  strides=(1, 1))(db_4)
    outputs = K.layers.Dense(units=1000,
                             activation="softmax",
                             kernel_initializer=initializer)(average_pooling2d)
    model = K.Model(inputs=inputs, outputs=outputs)
    return model
