#!/usr/bin/env python3
"""Creates a layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """prev is the previous layer, n is the number of nodes,
    activation is the activation"""
    he_et_al = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=he_et_al)
    return layer(prev)
