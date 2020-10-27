#!/usr/bin/env python3
"""Find the mean and covariance of a matrix"""
import numpy as np


def mean_cov(X):
    """Mean and covariance.
    Looked at how np does cov and found it to be fastest"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=1)
    N = X.shape[0]

    X -= mean[0]
    X_T = X
    c = np.matmul(X.T, X_T) / N

    return mean, c.squeeze()
