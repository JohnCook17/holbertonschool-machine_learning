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
            ei[sigma == 0.0] = 0.0
        X_next = self.X_s[(np.argmax(ei, axis=0))]
        return X_next, ei

    def optimize(self, iterations=100):
        """"""
        for i in range(iterations):
            print(i)
            X_next, ei = self.acquisition()
            Y_next = self.f(X_next)
            print(X_next, "\n", self.X_s)
            if X_next == self.X_s.any():
                if self.minimize:
                    return np.min(self.X_s), np.min(ei)
                else:
                    return np.max(self.X_s), np.max(ei)


            self.gp.update(X_next, Y_next)
            # use expected improvment to find next best point, get next best point, 
            # then get new y value, run the black box function to get y next so f,
            # sample f as little as possible,
            # update gp
            # if in X_s stop early
            # pick min or max y and coresponding x
        if self.minimize:
            return np.min(self.X_s), np.min(ei)
        else:
            return np.max(self.X_s), np.max(ei)

