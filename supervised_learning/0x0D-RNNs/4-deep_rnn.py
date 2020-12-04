#!/usr/bin/env python3
""""""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """"""
    H = []
    Y = []
    h = np.zeros(h_0.shape)
    H.append(h.copy())
    for time_step in range(X.shape[0]):
        for depth, layer in enumerate(rnn_cells):
            if depth == 0:
                temp_h, y = layer.forward(h[depth], X[time_step]) 
            else:
                temp_h, y = layer.forward(h[depth], h[depth - 1])
            h[depth] = temp_h
        Y.append(y)
        H.append(h.copy())
        h = np.zeros(h_0.shape)
    return np.asarray(H), np.asarray(Y)
