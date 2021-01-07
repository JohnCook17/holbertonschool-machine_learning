#!/usr/bin/env python3
"""SDP attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Scaled Dot Product Attention"""

    attn = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    attn = attn / tf.math.sqrt(dk)

    if mask is not None:
        attn += (mask * -1e9)

    attn = tf.keras.layers.Softmax(axis=-1)(attn)

    output = tf.matmul(attn, V)

    return output, attn
