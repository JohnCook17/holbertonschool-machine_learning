#!/usr/bin/env python3
""""""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """"""
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(units=n, activation=activation,
                                kernel_initializer=i)
    dropout = tf.layers.Dropout(new_layer, keep_prob)(prev)
    return dropout
