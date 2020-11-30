#!/usr/bin/env python3
"""variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """The encoder"""
    # encoder below
    inputs = keras.Input(shape=(input_dims,))

    enc = inputs

    for layer_dims in hidden_layers:
        enc = keras.layers.Dense(units=layer_dims, activation="relu")(enc)

    mean = keras.layers.Dense(units=latent_dims)(enc)
    log_sigma = keras.layers.Dense(units=latent_dims)(enc)

    def sampling(args):
        """the sampling function"""
        mean, log_sigma = args

        epsilon = keras.backend.random_normal(shape=(keras.backend
                                                     .shape(mean)[0],
                                                     latent_dims),
                                              mean=0, stddev=0.1)

        return mean + keras.backend.exp(log_sigma) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,)
                            )([mean, log_sigma])

    # decoder below
    dec_input = keras.Input(shape=(latent_dims,))
    dec = dec_input

    for layer_dims in reversed(hidden_layers):
        dec = keras.layers.Dense(units=layer_dims, activation="relu")(dec)

    dec_output_layer = keras.layers.Dense(units=input_dims,
                                          activation="sigmoid")(dec)
    # define encoder
    encoder = keras.Model(inputs=inputs, outputs=[z, mean, log_sigma])
    # define decoder

    decoder = keras.Model(inputs=dec_input, outputs=dec_output_layer)
    # define the autoencoder
    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=outputs)

    def custom_loss(inputs, outputs, input_dims, log_sigma, mean):
        """custom loss function"""

        def loss(inputs, outputs):
            """loss of the custom loss"""
            rec_loss = keras.losses.binary_crossentropy(inputs, outputs)
            rec_loss *= input_dims
            kl_loss = (1 + log_sigma - keras.backend
                       .square(mean) - keras.backend.exp(log_sigma))
            kl_loss = keras.backend.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = keras.backend.mean(rec_loss + kl_loss)
            return vae_loss
        return loss

    opt = keras.optimizers.Adam()

    loss = "binary_crossentropy"

    encoder.compile(loss=loss, optimizer=opt)
    decoder.compile(loss=loss, optimizer=opt)
    auto.compile(loss=custom_loss(inputs,
                                  outputs,
                                  input_dims,
                                  log_sigma,
                                  mean),
                 optimizer=opt)

    # encoder.summary()
    # decoder.summary()
    # auto.summary()

    return encoder, decoder, auto
