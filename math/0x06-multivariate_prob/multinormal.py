#!/usr/bin/env python3
"""Init of MultiNormal class"""
import numpy as np


class MultiNormal():
    """MultiNormal class"""
    def __init__(self, data):
        """inits self and data"""
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        def mean_cov(X):
            """Mean and covariance.
            Looked at how np does cov and found it to be fastest"""
            if not isinstance(X, np.ndarray):
                raise TypeError("X must be a 2D numpy.ndarray")
            if len(X.shape) != 2:
                raise TypeError("X must be a 2D numpy.ndarray")
            if X.shape[1] < 2:
                raise ValueError("X must contain multiple data points")

            X = X.T
            mean = np.mean(X, axis=0, keepdims=1)
            N = X.shape[0]
            X -= mean[0]
            X_T = X
            c = np.matmul(X.T, X_T) / N

            # print(mean, "\n", c.squeeze())

            return mean, c.squeeze()

        self.mean, self.cov = mean_cov(data)

    def pdf(self, x):
        """finds the pdf of a multivariate normal distribution"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = x.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({d}, 1)".format(d))

        mean = self.mean[0]
        cov = self.cov
        # print(mean, "\n", cov, "\n")

        pdf = (1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov)) *
               np.exp(-1/2 * (x - mean).T * np.linalg.inv(cov) * (x - mean)))
        pdf = np.sum(pdf) / d
        return pdf
