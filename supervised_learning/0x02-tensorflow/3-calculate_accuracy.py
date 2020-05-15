#!/usr/bin/env python3
"""tf accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """uses tf.reduce_mean, y is the lables, y_pred is the predictions"""
    pred = tf.argmax(y_pred, 1)
    equal = tf.equal(pred, y)
    return tf.reduce_mean(tf.cast(equal))
