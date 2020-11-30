#!/usr/bin/env python3
"""Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """uses convolution to autoencode"""
    inputs = keras.Input((input_dims))
    enc = inputs

    for fil in filters:
        enc = keras.layers.Conv2D(fil,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"
                                  )(enc)
        enc = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        padding="same"
                                        )(enc)

    bot_neck = enc

    dec_inputs = keras.Input((latent_dims))
    dec = dec_inputs

    last_filter = filters[-1]
    print(last_filter)
    filters = filters[:-1]
    print(filters)

    for fil in reversed(filters):
        dec = keras.layers.Conv2D(fil,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="relu"
                                  )(dec)
        dec = keras.layers.UpSampling2D((2, 2))(dec)

    second_to_last = keras.layers.Conv2D(last_filter,
                                         kernel_size=(3, 3),
                                         padding="valid",
                                         activation="relu"
                                         )(dec)
    second_to_last = keras.layers.UpSampling2D((2, 2))(second_to_last)
    outputs = keras.layers.Conv2D(input_dims[-1],
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="sigmoid"
                                  )(second_to_last)

    encoder = keras.Model(inputs=inputs, outputs=enc)
    decoder = keras.Model(inputs=dec_inputs, outputs=outputs)

    auto_input = keras.Input(shape=input_dims)
    enc_in = encoder(auto_input)
    dec_out = decoder(enc_in)
    auto = keras.Model(inputs=auto_input, outputs=dec_out)

    loss = "binary_crossentropy"
    opt = keras.optimizers.Adam()

    encoder.compile(loss=loss, optimizer=opt)
    decoder.compile(loss=loss, optimizer=opt)
    auto.compile(loss=loss, optimizer=opt)

    encoder.summary()
    decoder.summary()
    auto.summary()

    return encoder, decoder, auto
