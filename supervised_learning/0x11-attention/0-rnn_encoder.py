#!/usr/bin/env python3
"""An RNN Encoder"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """An RNN Encoder which inherits form Layers"""
    def __init__(self, vocab, embedding, units, batch):
        """Init the RNN"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Init hidden state"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """Calls the RNN encoder"""
        input_seq_len = tf.keras.backend.shape(x)[1]
        emb = self.embedding(x)
        outputs, hidden = self.gru(emb, initial_state=initial)

        return outputs, hidden
