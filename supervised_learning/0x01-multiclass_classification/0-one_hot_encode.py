#!/usr/bin/env python3
"""One hot encoding of the numbers 0 - 9"""
import numpy as np


def one_hot_encode(Y, classes):
    """One hot encodes using np"""
    if not isinstance(Y, np.ndarray):
        return None
    if not isinstance(classes, int):
        return None
    if Y.size == 0:
        return None
    if classes < 2:
        return None
    if classes < np.max(Y):
        return None
    one_hot = []
    for value in Y:
        number = [0 for n in range(classes) if isinstance(n, int)]
        number[value] = 1
        one_hot.append(number)
    return np.array(np.array(one_hot, dtype=float).T)
