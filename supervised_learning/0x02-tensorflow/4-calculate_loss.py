#!/usr/bin/env python3
"""Loss with tf"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """"""
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)
