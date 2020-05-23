#!/usr/bin/env python3
"""Batch normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """prev is the previous layer, n is the shape of the new layer,
    and activation is the activation"""
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    input_t = tf.layers.Dense(units=n, activation=activation,
                              kernel_initializer=i)(prev)
    mean, variance = tf.nn.moments(input_t, [0])
    return tf.nn.batch_normalization(input_t, mean=mean, variance=variance,
                                     offset=beta, scale=gamma,
                                     variance_epsilon=1e-8)
