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
        query = tf.expand_dims(s_prev, axis=1)

        score = self.V(tf.nn.tanh(self.W(query) + self.U(hidden_states)))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
        