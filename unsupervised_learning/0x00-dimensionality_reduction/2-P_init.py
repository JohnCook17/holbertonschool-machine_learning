#!/usr/bin/env python3
"""Init Tsne and the appropriate values"""
import numpy as np


def P_init(X, perplexity):
    """Initializes the values D, P, betas, and H"""
    n, d = X.shape

    def dist(X):
        """Finds the dist D"""
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)

        np.fill_diagonal(D, 0)

        return D

    def entropy():
        """Finds the shannon entropy H"""
        return np.log2(perplexity)

    D = dist(X)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = entropy()

    return D, P, betas, H
