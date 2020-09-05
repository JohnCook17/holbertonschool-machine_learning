#!/usr/bin/env python3
"""A Deep Recurant Neural Network"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Takes a list of RNN cells and runs forward pass on them"""
    Y = []
    H = []
    for l, rnn in enumerate(rnn_cells):
        h = h_0[l]
        h_ret = [h_0[l]]
        y_ret = []
        for i in range(X.shape[0]):
            print(i)
            h, y = rnn.forward(h, X[i])
            h_ret.append(h)
            y_ret.append(y)
        H.append(h_ret)
        Y.append(y_ret)
    return np.asarray(H), np.asarray(Y)
