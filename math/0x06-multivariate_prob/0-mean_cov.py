#!/usr/bin/env python3
""""""
import numpy as np


def mean_cov(X):
    """"""
    if not isinstance(X, np.ndarray) and len(X.shape) == 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    return np.mean(X, axis=0, keepdims=True), np.cov(X, rowvar=False)
