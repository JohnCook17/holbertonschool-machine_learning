#!/usr/bin/env python3
"""l2 initalization of weights for a new layer"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """uses he et al for init weight value, then uses l2 normalization"""
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    new_layer = tf.layers.Dense(units=n, activation=activation,
                                kernel_initializer=i,
                                kernel_regularizer=reg)
    return new_layer(prev)
