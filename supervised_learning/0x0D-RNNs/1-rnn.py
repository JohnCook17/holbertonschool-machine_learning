#!/usr/bin/env python3
""""""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """"""
    H = []
    Y = []
    H.append(h_0)
    h = h_0
    for time_step in range(X.shape[0]):
        h, y = rnn_cell.forward(h, X[time_step])
        H.append(h)
        Y.append(y)

    return np.asarray(H), np.asarray(Y)
