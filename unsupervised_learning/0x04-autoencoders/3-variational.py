#!/usr/bin/env python3
"""variational auto encoder"""
import tensorflow.keras as keras


class KLDivergenceLayer(keras.layers.Layer):
    """Adds kl loss to model loss"""
    def __init__(self, *args, **kwargs):
        """init of layer"""
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """what the layer does"""
        mu, log_var = inputs

        kl_batch = - 0.5 * keras.backend.sum(1 + log_var -
                                             keras.backend.square(mu) -
                                             keras.backend.exp(log_var),
                                             axis=1)

        self.add_loss(keras.backend.mean(kl_batch), inputs=inputs)

        return inputs


def encoder(input_dims, hidden_layers, latent_dims):
    """encodes the data"""
    inputs = keras.layers.Input(shape=(input_dims,))
    encoder = inputs
    for i in range(len(hidden_layers)):
        encoder = keras.layers.Dense(units=hidden_layers[i], activation="relu"
                                     )(encoder)
    # mean log_var layers
    mean = keras.layers.Dense(latent_dims)(encoder)
    log_var = keras.layers.Dense(latent_dims)(encoder)
    # KL divergence layer
    mean, log_var = KLDivergenceLayer()([mean, log_var])
    # std deviation layer
    std = keras.layers.Lambda(lambda t: keras.backend.exp(0.5 * t))(log_var)
    # epsilon and reconverge the layers
    eps = keras.layers.Input(tensor=keras.
                             backend.
                             random_normal(shape=(keras.
                                                  backend.
                                                  shape(inputs)[0],
                                                  latent_dims)))
    z_eps = keras.layers.Multiply()([std, eps])
    z = keras.layers.Add()([mean, log_var])
    # bottle neck
    bottle_neck = keras.layers.Dense(units=latent_dims, activation="relu"
                                     )(z)

    encoder_m = keras.Model(inputs=inputs,
                            outputs=[bottle_neck,
                                     mean,
                                     log_var])
    encoder_m.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder_m


def decoder(input_dims, hidden_layers, latent_dims):
    """decodes the data"""
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

    return decoder_m


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ties it all together"""
    inputs = keras.layers.Input(shape=(input_dims,))
    var_encoder_m = encoder(input_dims, hidden_layers, latent_dims)
    var_encoder = var_encoder_m(inputs)
    var_decoder_m = decoder(input_dims, hidden_layers, latent_dims)
    var_decoder = var_decoder_m(var_encoder)
    auto = keras.Model(inputs=inputs, outputs=var_decoder)
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return var_encoder_m, var_decoder_m, auto
