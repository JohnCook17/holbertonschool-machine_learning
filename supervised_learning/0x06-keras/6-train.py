#!/usr/bin/env python3
"""Uses .fit to train the network"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """network is the network to train, data is the data, labels is the data
    labels, batch size is the size of the minibatch, epochs is the number of
    epochs to run, verbose is print of not the progress, shuffle is to shuffle
    the data or not, now uses validation data as well. now with early stopping
    """
    if validation_data is not None:
        if early_stopping:
            my_callbacks = [K.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience)]
            return network.fit(x=data, y=labels,
                               validation_data=validation_data,
                               epochs=epochs, batch_size=batch_size,
                               verbose=verbose, shuffle=shuffle,
                               callbacks=my_callbacks)
        else:
            return network.fit(x=data, y=labels,
                               validation_data=validation_data,
                               epochs=epochs, batch_size=batch_size,
                               verbose=verbose, shuffle=shuffle)
    else:
        return network.fit(x=data, y=labels, epochs=epochs,
                           batch_size=batch_size,
                           verbose=verbose, shuffle=shuffle)
