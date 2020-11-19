#!/usr/bin/env python3
"""Bayesian Optimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """The Bayesian Optimization class"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """init of Bayesian Optimization"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = (np.linspace(start=bounds[0], stop=bounds[1],
                                num=ac_samples)[:, np.newaxis])
        self.xsi = xsi
        self.minimize = minimize
