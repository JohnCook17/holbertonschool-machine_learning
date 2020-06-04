#!/usr/bin/env python3
"""Uses .fit to train the network"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """network is the network to train, data is the data, labels is the data
    labels, batch size is the size of the minibatch, epochs is the number of
    epochs to run, verbose is print of not the progress, shuffle is to shuffle
    the data or not, now uses validation data as well. now with early stopping
    """
    def scheduler(epoch, alpha=alpha, decay_rate=decay_rate):
        """schedules the decay rate"""
        return alpha / (1 + decay_rate * (epoch // 1))

    if validation_data is not None:
        if early_stopping and learning_rate_decay:
            decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
            my_callbacks = [K.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience),
                            decay]
            return network.fit(x=data, y=labels,
                               validation_data=validation_data,
                               epochs=epochs, batch_size=batch_size,
                               verbose=verbose, shuffle=shuffle,
                               callbacks=my_callbacks)
        elif learning_rate_decay and not early_stopping:
            decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
            my_callbacks = [decay]
            return network.fit(x=data, y=labels,
                               validation_data=validation_data,
                               epochs=epochs, batch_size=batch_size,
                               verbose=verbose, shuffle=shuffle,
                               callbacks=my_callbacks)
        elif early_stopping and not learning_rate_decay:
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
