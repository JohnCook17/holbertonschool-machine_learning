#!/usr/bin/env python3
"""Gradient descent with momentum"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """loss is a tf loss tensor, alpha is the learning rate
    and beta1 is momentum"""
    return(tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
           .minimize(loss))
