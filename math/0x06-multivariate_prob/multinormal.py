#!/usr/bin/env python3
"""MultiNormal class"""
import numpy as np


def cov(x, y):
    """covariance"""
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar) * (y - ybar)) / (len(x) - 1)


def cov_mat(X):
    """finds the covariance matrix"""
    d, n = X.shape
    return np.array([[cov(X[i], X[j]) for j in range(d)] for i in range(d)])


class MultiNormal():
    """The MultiNormal class"""
    def __init__(self, data):
        """init"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data.T, axis=0, keepdims=True).T
        self.cov = cov_mat(data)
