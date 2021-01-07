#!/usr/bin/env python3
"""SDP attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Scaled Dot Product Attention"""

    attn = tf.matmul(Q, K, transpose_b=True)

    if mask is not None:
        attn = tf.add(tf.multiply(mask(attn), -1e9), attn)
    
    attn = tf.keras.layers.Softmax(axis=-1)(attn)

    output = tf.matmul(attn, V)

    return output, attn
