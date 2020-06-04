#!/usr/bin/env python3
"""Uses .fit to train the network"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """network is the network to train, data is the data, labels is the data
    labels, batch size is the size of the minibatch, epochs is the number of
    epochs to run, verbose is print of not the progress, shuffle is to shuffle
    the data or not, now uses validation data as well. now with early stopping
    """
    def scheduler(epoch):
        """schedules the decay rate"""
        return (alpha / (1 + decay_rate * epoch))
    if validation_data is not None:
        my_callbacks = []
        if save_best:
            my_callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                            monitor="val_loss",
                                                            save_best_only=True
                                                            ))
        if early_stopping:
            my_callbacks.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=patience,
                                                          mode="min"))
        if learning_rate_decay:
            my_callbacks.append(K.callbacks.LearningRateScheduler(scheduler,
                                                                  verbose=1))
        return network.fit(x=data, y=labels,
                           validation_data=validation_data,
                           epochs=epochs, batch_size=batch_size,
                           verbose=verbose, shuffle=shuffle,
                           callbacks=my_callbacks)
    else:
        return network.fit(x=data, y=labels, epochs=epochs,
                           batch_size=batch_size,
                           verbose=verbose, shuffle=shuffle)
