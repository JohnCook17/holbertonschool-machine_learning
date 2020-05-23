#!/usr/bin/env python3
"""Batch normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """prev is the previous layer, n is the shape of the new layer,
    and activation is the activation"""
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None, kernel_initializer=i)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    norm = tf.nn.batch_normalization(layer(prev), mean=mean, variance=variance,
                                     offset=beta, scale=gamma,
                                     variance_epsilon=1e-8)
    return activation(norm)
