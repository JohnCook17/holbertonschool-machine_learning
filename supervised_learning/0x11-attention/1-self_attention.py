#!/usr/bin/env python3
"""Self Attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self Attention which inherits form Layer"""
    def __init__(self, units):
        """Init of Self Attention"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Calls the Self Attention layers"""
        query = tf.expand_dims(s_prev, axis=1)

        score = self.V(tf.nn.tanh(self.W(query) + self.U(hidden_states)))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
        