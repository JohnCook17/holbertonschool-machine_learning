#!/usr/bin/env python3
"""Creates placeholders for tf"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """x is the input data placeholder,
    y is for the onehot lables for input data"""
    x = tf.placeholder(float, [None, nx], name="x")
    y = tf.placeholder(float, [None, classes], name="y")
    return x, y
