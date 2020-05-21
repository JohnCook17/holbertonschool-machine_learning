#!/usr/bin/env python3
"""Shuffles data using np.random.permutation"""
import numpy as np


def shuffle_data(X, Y):
    """X is the first np.ndarray to shuffle. Y is the second, both have a shape
    of (m, nx), and (m, ny) respectivly m being the number of data points
    and the n value being the number of features in the respective letter
    returns shuffled x and y"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
