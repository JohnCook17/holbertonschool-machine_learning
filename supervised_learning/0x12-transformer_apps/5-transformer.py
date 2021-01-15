#!/usr/bin/env python3
"""The Transformer Model"""
import tensorflow as tf


class Transformer(tf.keras.Model):
    """The Transformer model class"""
    def __init__(self,
                 N,
                 dm,
                 h,
                 hidden,
                 input_vocab,
                 target_vocab,
                 max_seq_input,
                 max_seq_target,
                 encoder,
                 decoder,
                 drop_rate=0.1):
        """init of transformer"""
        super(Transformer, self).__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.linear = tf.keras.layers.Dense(units=target_vocab,
                                            input_shape=([None], [None]))

    def call(self,
             inputs,
             target,
             training,
             encoder_mask,
             look_ahead_mask,
             decoder_mask):
        """Calls the transformer"""
        print("============================")
        print(inputs)
        print(target)
        print("============================")
        # pt_output, en_output = self.encoder(inputs, target)

        dec_output = self.decoder.decode(ids=target.numpy().squeeze().tolist())

        dec_output = dec_output[:, tf.newaxis]

        final_output = self.linear(dec_output)

        print("============================")
        print(type(final_output))
        print("============================")

        return final_output
