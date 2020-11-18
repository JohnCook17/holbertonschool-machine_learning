#!/usr/bin/env python3
"""The Gaussian Process used in optimization"""
import numpy as np


class GaussianProcess():
    """The Gaussian Process Class"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """init of variables used in the gp"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.Y)

    def kernel(self, X1, X2):
        """The Radial Basis Function to get the kernel"""
        return ((self.sigma_f ** 2) *
                np.exp(-(1 / (2 * (self.l ** 2))) * ((X1 - X2).T * (X1 - X2))))
