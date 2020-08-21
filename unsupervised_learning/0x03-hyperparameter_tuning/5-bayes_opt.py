#!/usr/bin/env python3
"""Bayesian optimization, using the Gaussian process"""
import numpy as np
from scipy.stats import norm
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
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """the acquisition function"""
        mu, sigma = self.gp.predict(self.X_s)
        # sigma = sigma.reshape(-1, 1)

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        with np.errstate(divide="warn"):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[np.isclose(sigma, 0.0)] = 0.0
        X_next = self.X_s[(np.argmax(ei, axis=0))]
        return X_next, ei

    def optimize(self, iterations=100):
        """optimizes X based on best Y"""
        for i in range(iterations):
            X_next, ei = self.acquisition()
            Y_next = self.f(X_next)
            if np.isin(X_next, self.gp.X):
                if self.minimize:
                    Y_opt = np.min(self.gp.Y, keepdims=True)
                    X_opt = self.gp.X[np.argmin(self.gp.Y)]
                    return X_opt, Y_opt[0]
                else:
                    Y_opt = np.max(self.gp.Y, keepdims=True)
                    X_opt = self.gp.X[np.argmax(self.gp.Y)]
                    return X_opt, Y_opt[0]

            self.gp.update(X_next, Y_next)
        if self.minimize:
            Y_opt = np.min(self.gp.Y, keepdims=True)
            X_opt = self.gp.X[np.argmin(self.gp.Y)]
            return X_opt, Y_opt[0]
        else:
            Y_opt = np.max(self.gp.Y, keepdims=True)
            X_opt = self.gp.X[np.argmax(self.gp.Y)]
            return X_opt, Y_opt[0]
