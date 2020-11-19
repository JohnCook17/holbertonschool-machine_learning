#!/usr/bin/env python3
"""Bayesian Optimization"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """"""
        mu, sigma = self.gp.predict(self.X_s)
        mu_sample, _ = self.gp.predict(self.gp.X)

        # sigma = sigma.reshape(-1, 1)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        imp = mu - mu_sample_opt - self.xsi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[ei == 0.0] = 0.0

        return None, ei
