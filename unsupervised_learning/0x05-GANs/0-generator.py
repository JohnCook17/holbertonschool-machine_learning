#!/usr/bin/env python3
"""The GANs generator"""
import tensorflow as tf


def generator(Z):
    """The GANs generator function"""
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        X = tf.layers.Dense(units=128, activation="relu", name="layer_1")(Z)
        X = tf.layers.Dense(units=784, activation="sigmoid", name="layer_2")(X)
        return X
