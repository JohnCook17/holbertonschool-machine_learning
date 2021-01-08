#!/usr/bin/env python3
"""The Transformer Model"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


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
                 drop_rate=0.1):
        """init of transformer"""
        super(Transformer, self).__init__()

        self.encoder = Encoder(N,
                               dm,
                               h,
                               hidden,
                               input_vocab,
                               max_seq_input,
                               drop_rate)

        self.decoder = Decoder(N,
                               dm,
                               h,
                               hidden,
                               target_vocab,
                               max_seq_target,
                               drop_rate)

        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self,
             inputs,
             target,
             training,
             encoder_mask,
             look_ahead_mask,
             decoder_mask):
        """Calls the transformer"""
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output = self.decoder(target,
                                  enc_output,
                                  training,
                                  look_ahead_mask,
                                  decoder_mask)

        final_output = self.linear(dec_output)

        return final_output
