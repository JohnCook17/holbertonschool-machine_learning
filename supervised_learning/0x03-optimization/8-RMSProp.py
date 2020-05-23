#!/usr/bin/env python3
"""RSMProp in tf"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """loss is a tf loss tensor. alpha is the learnign rate.
    beta2 is the decay rate, epsilon avoids zero division"""
    return tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)
