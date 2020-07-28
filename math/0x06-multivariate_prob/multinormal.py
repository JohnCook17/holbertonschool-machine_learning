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
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data.T, axis=0, keepdims=True).T
        self.cov = cov_mat(data)

    def pdf(self, x):
        """Finds the pdf of a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must by a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError("x must have the shape({}, 1)".format(x.shape[0]))
        k = x.shape[0]
        con = 1 / ((2 * np.pi) ** (k / 2) * np.linalg.det(self.cov) ** 0.5)
        exp = np.exp(-0.5 * np.matmul(np.matmul((x - self.mean).T,
                                                np.linalg.inv(self.cov)),
                                      (x - self.mean)))
        x_hat = con * exp
        return x_hat[0, 0]
