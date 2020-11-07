#!/usr/bin/env python3
"""Gaussian Mixture Model in sklearn"""
import sklearn.mixture


def gmm(X, k):
    """The gaussian mixture model"""
    gauss_mix = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = gauss_mix.weights_
    m = gauss_mix.means_
    S = gauss_mix.covariances_
    clss = gauss_mix.predict(X)
    bic = gauss_mix.bic(X)

    return pi, m, S, clss, bic
