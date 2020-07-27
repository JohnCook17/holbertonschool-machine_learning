#!/usr/bin/env python3
"""Mean and Covariance of a matrix"""
import numpy as np


def cov(x, y):
    """covariance"""
    xbar, ybar= x.mean(), y.mean()
    return np.sum((x - xbar) * (y - ybar)) / (len(x) - 1)


def cov_mat(X):
    """"""
    d, n = X.shape
    return np.array([[cov(X[i], X[j]) for j in range(d)] for i in range(d)])


def mean_cov(X):
    """Returns the mean and covariance"""
    if not isinstance(X, np.ndarray) and len(X.shape) == 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    return np.mean(X, axis=0, keepdims=True), cov_mat(X.T)
