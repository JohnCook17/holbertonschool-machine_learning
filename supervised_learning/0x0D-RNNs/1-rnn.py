#!/usr/bin/env python3
"""Runs an Rnn cell a number of times"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """runs rnn on data X,
    h_0 is the initial hidden state,
    rnn_cell is the cell to use"""
    h = h_0
    h_ret = [h_0]
    y_ret = []
    for i in range(X.shape[0]):
        h, y = rnn_cell.forward(h, X[i])
        h_ret.append(h)
        y_ret.append(y)
    return np.asarray(h_ret), np.asarray(y_ret)
