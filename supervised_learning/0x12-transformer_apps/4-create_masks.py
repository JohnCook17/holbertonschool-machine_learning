#!/usr/bin/env python3
"""Mask so that null values of padding are not accounted"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Creates 3 mask"""

    def create_padding_mask(seq):
        """creates a padding mask"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        return seq[: tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(size):
        """creates la mask"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    enc_mask = create_padding_mask(inputs)

    dec_mask = create_padding_mask(inputs)

    la_mask = create_la_mask(tf.shape(target)[1])
    dec_tar_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_tar_mask, la_mask)

    return enc_mask, combined_mask, dec_mask
