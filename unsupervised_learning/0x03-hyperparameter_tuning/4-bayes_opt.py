#!/usr/bin/env python3
"""Bayesian optimization, using the Gaussian process"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """The Bayesian Optimization class"""
    def __init__(self,
                 f,
                 X_init,
                 Y_init,
                 bounds,
                 ac_samples,
                 l=1,
                 sigma_f=1,
                 xsi=0.01,
                 minimize=True):
        """Init values for later use"""
        self.f = f
        self.gp = GP(X_init, Y_init)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)[:, np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """"""
        mu, sigma = self.gp.predict(self.X_s)
        mu_sample = self.gp.predict(self.gp.X)
        # sigma = sigma.reshape(-1, 1)
        mu_max = np.max(mu_sample)
        with np.errstate(divide="warn"):
            imp = mu - mu_max - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return mu_max, ei
