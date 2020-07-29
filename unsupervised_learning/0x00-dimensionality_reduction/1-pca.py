#!/usr/bin/env python3
"""uses svd to get pca"""
import numpy as np


def pca(X, ndim):
    """finds the w of pca"""
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    w = vh.T
    T = np.matmul(X_m, w[:, 0: ndim])
    return T
