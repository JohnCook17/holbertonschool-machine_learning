#!/usr/bin/env python3
"""Transformer Decoder"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """The Decoder for the transformer consisting of N blocks"""
    def __init__(self,
                 N,
                 dm,
                 h,
                 hidden,
                 target_vocab,
                 max_seq_len,
                 drop_rate=0.1):
        """"""
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Calls N blocks of decoder"""
        seq_len = x.get_shape()[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:self.dm, :seq_len, None].T

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x,
                               encoder_output,
                               training,
                               look_ahead_mask,
                               padding_mask)

        return x
