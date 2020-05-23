#!/usr/bin/env python3
"""ADAM in tf"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """loss is a loss tensor. alpha is the learning rate.
    beta1 is the weight used for the first moment. beta2
    is the weight used for the second moment. epsilon avoids
    zero division."""
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)
