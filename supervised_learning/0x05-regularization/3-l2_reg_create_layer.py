#!/usr/bin/env python3
""""""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """"""
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    return tf.layers.dense(inputs=prev, units=n, activation=activation,
                           kernel_regularizer=reg)
