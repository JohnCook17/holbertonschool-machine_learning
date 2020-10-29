#!/usr/bin/env python3
"""Performs PCA on a matrix"""
import numpy as np


def pca(X, ndim):
    """Returns T after performing PCA on a matrix"""
    X = X - np.mean(X, axis=0)
    V = np.linalg.svd(X)
    V = V[2]
    T = np.matmul(X, V[:ndim].T)
    return T
