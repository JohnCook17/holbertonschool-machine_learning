#!/usr/bin/env python3
""""""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """"""
    H = h_0
    H_o = np.zeros((X.shape[0] + 1, h_0.shape[0], h_0.shape[1], h_0.shape[2]))
    print(H_o.shape)
    Y = []
    for time_step in range(X.shape[0]):
        h = None
        for depth, layer in enumerate(rnn_cells):
            # print(time_step, depth)
            if h is None:
                h, y = layer.forward(H[depth], X[time_step])
            else:
                h, y = layer.forward(H[depth], h)
            # print(H.shape, h.shape)
            H[depth, :, :] = h
        H_o[time_step + 1, :, :, :] = H
        H = np.zeros(h_0.shape)
        Y.append(y)
    return H_o, np.asarray(Y)
