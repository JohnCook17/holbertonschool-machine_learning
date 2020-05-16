#!/usr/bin/env python3
"""Train op using tf"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Trains the neural network using gradient descent minimizing loss"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
