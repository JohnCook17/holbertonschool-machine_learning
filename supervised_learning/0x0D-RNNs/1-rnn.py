#!/usr/bin/env python3
""""""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """"""
    H = np.zeros(shape=(X.shape[0] + 1, h_0.shape[0], h_0.shape[1]))
    Y = []
    for time_step in range(X.shape[0]):
        h, y = rnn_cell.forward(h_0, X[time_step])
        H[time_step + 1:, :, :] = h
        Y.append(y)

    return np.asarray(H), np.asarray(Y)
