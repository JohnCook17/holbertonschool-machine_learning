#!/usr/bin/env python3
""""""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """"""
    hf = h_0
    hb = h_t
    h_ret = []
    Y = []
    for t in range(X.shape[0]):
        hf = bi_cell.forward(hf, X[t])
        hb = bi_cell.backward(hb, X[X.shape[0] - 1 - t])
        hs = np.concatenate((hf, hb), axis=1)
        y = bi_cell.output(hs)
        h_ret.append(hs)
        Y.append(y)
    return np.asarray(h_ret), np.asarray(Y)
