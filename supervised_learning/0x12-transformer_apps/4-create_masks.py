#!/usr/bin/env python3
"""Mask so that null values of padding are not accounted"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Creates 3 mask"""
    enc_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    enc_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

    dec_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    dec_mask = dec_mask[:, tf.newaxis, tf.newaxis, :]

    la_size = tf.shape(target)[1]
    la_mask = 1 - tf.linalg.band_part(tf.ones((la_size,
                                               la_size)), -1, 0)

    tar_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    tar_mask = tar_mask[:, tf.newaxis, tf.newaxis, :]

    cmb_mask = tf.maximum(tar_mask, la_mask)

    return enc_mask, cmb_mask, dec_mask
