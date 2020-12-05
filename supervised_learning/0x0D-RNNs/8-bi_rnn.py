#!/usr/bin/env python3
"""A Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Forward prop"""
    H = []
    H_f = []
    H_b = []
    Y = []
    h_f = h_0
    h_b = h_t
    for time_step in range(X.shape[0]):
        h_f = bi_cell.forward(h_f, X[time_step])
        h_b = bi_cell.backward(h_b, X[time_step])
        H_f.append(h_f)
        H_b.append(h_b)
        
    H = np.concatenate((H_f, H_b), axis=2)
    # print(H.shape)

    for h in H:
        y = bi_cell.output(h)
        Y.append(y)

    return np.asarray(H), np.asarray(Y)
