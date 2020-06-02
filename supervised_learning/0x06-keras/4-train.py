#!/usr/bin/env python3
""""""


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """"""
    return network.fit(x=data, y=labels, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, shuffle=shuffle)
