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
        """Acquires X_next, and Expectation improvement"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        with np.errstate(divide="warn"):
            imp = mu_sample_opt - mu - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[ei == 0.0] = 0
        X = self.X_s[np.argmax(ei, axis=0)]
        return X, ei

    def optimize(self, iterations=100):
        """The optimization step"""
        X_searched = []
        for _ in range(iterations):
            X, _ = self.acquisition()
            Y = self.f(X)
            if X in X_searched:
                break
            X_searched.append(X)
            self.gp.update(X, Y)
        np.delete(self.gp.X[:, -1])
        if self.minimize:
            Y_opt = np.min(self.gp.Y, keepdims=True)
            return self.gp.X[np.argmin(self.gp.Y)], Y_opt[0]
        else:
            Y_opt = np.max(self.gp.Y, keepdims=True)
            return self.gp.X[np.argmax(self.gp.Y)], Y_opt[0]
