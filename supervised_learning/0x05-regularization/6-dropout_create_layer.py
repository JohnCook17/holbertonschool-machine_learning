#!/usr/bin/env python3
"""Creates a new layer in tf with dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creating a new layer with dropout"""
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(units=n, activation=activation,
                                kernel_initializer=i)(prev)
    dropout = tf.layers.Dropout(rate=keep_prob)(new_layer)
    return dropout
