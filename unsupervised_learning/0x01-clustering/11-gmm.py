#!/usr/bin/env python3
"""GMM in sklearn"""
import sklearn.mixture


def gmm(X, k):
    """GMM using sklearn where X is the data and K is num clusters"""
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
