#!/usr/bin/env python3
""""""
import numpy as np


class GaussianProcess:
    """"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        sqdist = (np.sum(self.X ** 2, 1).reshape(-1, 1) +
                  np.sum(self.X ** 2, 1) - 2 * np.matmul(self.X, self.X.T))
        self.K = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def kernel(self, X1, X2):
        """"""
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
                  np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T))
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
