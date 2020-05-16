#!/usr/bin/env python3
"""Loss with tf"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """uses softmax to calculate loss, with y lables and y_pred
    predictions"""
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)
