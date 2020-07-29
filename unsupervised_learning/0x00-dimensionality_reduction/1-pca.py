#!/usr/bin/env python3
"""uses svd to get pca"""
import numpy as np


def pca(X, ndim):
    """finds the w of pca"""
    u, s, vh = np.linalg.svd(X)
    w = vh.T
    X_m = X - np.mean(X, axis=0)
    T = np.matmul(X_m, w[:, 0: ndim])
    return T
