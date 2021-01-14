#!/usr/bin/env python3
"""Mask so that null values of padding are not accounted"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Creates 3 mask"""
    enc_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

    la_size = tf.size(target)
    la_mask = 1 - tf.linalg.band_part(tf.ones((la_size, la_size)), -1, 0)

    dec_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    dec_mask = dec_mask[:, tf.newaxis, tf.newaxis, :]

    return enc_mask, la_mask, dec_mask
