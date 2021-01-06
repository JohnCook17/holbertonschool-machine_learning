#!/usr/bin/env python3
""""""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """"""
    def __init__(self, vocab, embedding, units, batch):
        """"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=embedding, output_dim=self.units)
        self.gru = tf.keras.layers.GRU(units=units, return_state=True)

    def initialize_hidden_state(self):
        """"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """"""
        input_seq_len = tf.keras.backend.shape(x)[1]
        # emb = self.embedding(x)
        # x = tf.expand_dims(x, axis=0)
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb)

        return outputs, hidden
