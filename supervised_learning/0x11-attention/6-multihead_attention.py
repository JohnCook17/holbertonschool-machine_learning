#!/usr/bin/env python3
""""""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """"""
    def __init__(self, dm, h):
        """"""
        super().__init__()

        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """"""
        batch = Q.shape[0]
        seq_len_q = Q.shape[1]
        seq_len_v = K.shape[1]
        dk = K.shape[2]
        dv = V.shape[2]

        residual = Q

        Q = self.Wq(Q)  # tf.reshape(, (batch, seq_len_q, self.h, dk))
        K = self.Wk(K)  # tf.reshape(, (batch, seq_len_v, self.h, dk))
        V = self.Wv(V)  # tf.reshape(, (batch, seq_len_v, self.h, dv))

        Q, attn = sdp_attention(Q, K, V, mask)

        Q = tf.concat((Q, residual), axis=2)

        Q = self.linear(Q)

        return Q, attn
