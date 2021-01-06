#!/usr/bin/env python3
""""""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """"""
    def __init__(self, units):
        """"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """"""
        s_prev_w = self.W(s_prev)
        hidden_states_u = self.U(hidden_states)

        context = tf.keras.backend.sum(s_prev_w, axis=0)
        weights = tf.keras.backend.sum(tf.keras.backend.dot(hidden_states_u, tf.keras.backend.transpose(s_prev_w)), axis=0, keepdims=True)
        weights = tf.keras.backend.transpose(weights)

        return context, weights
        