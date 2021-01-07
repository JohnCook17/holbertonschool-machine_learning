#!/usr/bin/env python3
"""A RNN Decoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """An RNN Decoder which inherits from Layer"""
    def __init__(self, vocab, embedding, units, batch):
        """"""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab)
        self.units = units

    def call(self, x, s_prev, hidden_states):
        """Calls the RNN encoder"""
        s = SelfAttention(self.units)
        context, weights = s.call(s_prev=s_prev, hidden_states=hidden_states)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context, axis=1), x], axis=2)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)

        return output, x
