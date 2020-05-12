#!/usr/bin/env python3
"""One hot encoding of the numbers 0 - 9"""
import numpy as np


def one_hot_encode(Y, classes):
    """One hot encodes using np"""
    if Y.size == 0:
        return None
    if classes < 1:
        return None
    one_hot = []
    for value in Y:
        number = [0 for n in range(classes) if isinstance(n, int)]
        number[value] = 1
        one_hot.append(number)
    return np.array(one_hot, dtype=float).T
