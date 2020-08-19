#!/usr/bin/env python3
"""Gaussian Process"""
import numpy as np


class GaussianProcess:
    """The Guassian Process optimization class"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """init values for later calculations"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        sqdist = (np.sum(self.X ** 2, 1).reshape(-1, 1) +
                  np.sum(self.X ** 2, 1) - 2 * np.matmul(self.X, self.X.T))
        self.K = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def kernel(self, X1, X2):
        """The kernel to use in later calculations"""
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
                  np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T))
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """Makes a prediction"""
        A = self.kernel(X_s, self.X)
        B = self.kernel(self.X, self.X)
        C = self.kernel(X_s, X_s)
        D = self.kernel(self.X, X_s)

        mu = A @ np.linalg.inv(B) @ self.Y
        sigma = C - (A @ np.linalg.inv(B) @ D)

        return mu.flatten(), np.diag(sigma)

    def update(self, X_new, Y_new):
        """updates the kernel and x and y"""
        self.X = np.append(self.X, X_new)[np.newaxis].T
        self.Y = np.append(self.Y, Y_new)[np.newaxis].T
        self.K = self.kernel(self.X, self.X)
