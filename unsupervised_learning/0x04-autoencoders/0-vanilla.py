#!/usr/bin/env python3
""""""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """"""
    input_layer = keras.Input(shape=(input_dims,))

    enc = input_layer

    for layer_dims in hidden_layers:
        enc = keras.layers.Dense(units=layer_dims, activation="relu")(enc)

    bot_neck = keras.layers.Dense(units=latent_dims, activation="relu")(enc)

    dec_input = keras.Input(shape=(latent_dims,))
    dec = dec_input

    for layer_dims in reversed(hidden_layers):
        dec = keras.layers.Dense(units=layer_dims, activation="relu")(dec)

    out_put = keras.layers.Dense(units=input_dims, activation="sigmoid")(dec)

    # define the 3 models
    encoder = keras.Model(inputs=input_layer, outputs=bot_neck)
    decoder = keras.Model(inputs=dec_input, outputs=out_put)
    # define the autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    enc_in = encoder(auto_input)
    dec_out = decoder(enc_in)
    auto = keras.Model(inputs=auto_input, outputs=dec_out)

    loss = "binary_crossentropy"
    opt = keras.optimizers.Adam()

    encoder.compile(loss=loss, optimizer=opt)
    decoder.compile(loss=loss, optimizer=opt)
    auto.compile(loss=loss, optimizer=opt)

    return encoder, decoder, auto
