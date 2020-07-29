#!/usr/bin/env python3
"""uses svd to get pca"""
import numpy as np


def pca(X, ndim):
    """finds the w of pca"""
    u, s, vh = np.linalg.svd(X)
    w = vh.T
    T = X @ w[0:, 0:ndim]
    return T
