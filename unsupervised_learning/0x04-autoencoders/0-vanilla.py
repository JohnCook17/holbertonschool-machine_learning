#!/usr/bin/env python3
"""vanilla auto encoder"""
# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """makes an encoder, a decoder, and a autoencoder"""

    # encoder
    inputs = keras.layers.Input(shape=(input_dims,))
    encoder = inputs
    for i in range(len(hidden_layers)):
        encoder = keras.layers.Dense(units=hidden_layers[i], activation="relu"
                                     )(encoder)

    bottle_neck_c = keras.layers.Dense(units=latent_dims, activation="relu"
                                       )(encoder)

    encoder_m = keras.Model(inputs=inputs, outputs=bottle_neck_c)
    encoder_m.compile(optimizer="adam", loss="binary_crossentropy")
    # encoder_m.summary()

    # decoder
    inputs_d = keras.layers.Input(shape=(latent_dims,))
    for i in range(-1, -(len(hidden_layers) + 1), -1):
        if i == -1:
            decoder_c = keras.layers.Dense(units=hidden_layers[i],
                                           activation="relu")(inputs_d)
        else:
            decoder_c = keras.layers.Dense(units=hidden_layers[i],
                                           activation="relu")(decoder_c)

    last_c = keras.layers.Dense(units=input_dims,
                                activation="sigmoid")(decoder_c)

    decoder_m = keras.Model(inputs=inputs_d, outputs=last_c)
    decoder_m.compile(optimizer="adam", loss="binary_crossentropy")

    # decoder_m.summary()

    # autoencoder
    auto_inputs = keras.layers.Input(shape=(input_dims,))
    encoder_out = encoder_m(auto_inputs)
    decoder_out = decoder_m(encoder_out)
    model = keras.Model(inputs=auto_inputs, outputs=decoder_out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    # model.summary()
    return encoder_m, decoder_m, model
