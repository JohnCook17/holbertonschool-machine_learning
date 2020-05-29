#!/usr/bin/env python3
"""l2 cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """l2 cost in tf"""
    l2cost = tf.nn.l2_loss(cost)
    return tf.sqrt(tf.math.divide(tf.math.reduce_sum(tf.square(cost)), 2))
