#!/usr/bin/env python3
"""Convolutional auto encoder"""
import tensorflow.keras as keras


def std_layer_encode(inputs, filter_size):
    """the standard layer for encoding"""
    conv = keras.layers.Conv2D(filters=filter_size,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="relu")(inputs)
    pool = keras.layers.MaxPool2D(pool_size=(2, 2), padding="same")(conv)
    return pool


def std_layer_decode(inputs, filter_size):
    """the standard layer for decoding"""
    conv = keras.layers.Conv2D(filters=filter_size,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="relu")(inputs)
    upsampleing = keras.layers.UpSampling2D(size=(2, 2))(conv)
    return upsampleing


def encoder(input_dims, filters, latent_dims):
    """encodes data"""
    inputs = keras.layers.Input(shape=input_dims)
    encode_layer = inputs
    for i in range(len(filters)):
        encode_layer = std_layer_encode(encode_layer, filters[i])

    model = keras.Model(inputs=inputs, outputs=encode_layer)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def decoder(input_dims, filters, latent_dims):
    """decodes data"""
    inputs = keras.layers.Input(shape=latent_dims)
    decoded_layer = inputs
    for i in range(-1, -(len(filters)), -1):
        decoded_layer = std_layer_decode(decoded_layer, filters[i])
    scnd_last = keras.layers.Conv2D(filters=filters[i],
                                    kernel_size=(3, 3),
                                    padding="valid",
                                    activation="relu")(decoded_layer)
    upsampleing = keras.layers.UpSampling2D(size=(2, 2))(scnd_last)
    last = keras.layers.Conv2D(filters=1,
                               kernel_size=(3, 3),
                               padding="same",
                               activation="sigmoid")(upsampleing)

    model = keras.Model(inputs=inputs, outputs=last)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def autoencoder(input_dims, filters, latent_dims):
    """ties it all together"""
    inputs = keras.layers.Input(shape=input_dims)
    cnn_encoder_m = encoder(input_dims, filters, latent_dims)
    cnn_encoder = cnn_encoder_m(inputs)
    cnn_decoder_m = decoder(input_dims, filters, latent_dims)
    cnn_decoder = cnn_decoder_m(cnn_encoder)
    auto = keras.Model(inputs=inputs, outputs=cnn_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return cnn_encoder_m, cnn_decoder_m, auto
