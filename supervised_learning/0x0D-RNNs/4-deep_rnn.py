#!/usr/bin/env python3
""""""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """"""
    H = np.zeros(shape=(X.shape[0] + 1, h_0.shape[0], h_0.shape[1], h_0.shape[2]))
    Y = []
    for time_step in range(X.shape[0]):
        for depth, layer in enumerate(rnn_cells):
            h, y = layer.forward(h_0[depth], X[time_step])
            H[time_step + 1:, depth, :, :] = h
        Y.append(y)

    return np.asarray(H), np.asarray(Y)
