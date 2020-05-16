#!/usr/bin/env python3
"""tf accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """uses tf.reduce_mean, y is the lables, y_pred is the predictions"""
    y_label = tf.argmax(y, 1)
    pred = tf.argmax(y_pred, 1)
    equal = tf.equal(pred, y_label)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
