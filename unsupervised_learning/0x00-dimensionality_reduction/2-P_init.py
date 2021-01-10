#!/usr/bin/env python3
""""""
import numpy as np


def P_init(X, perplexity):
    """"""
    n, d = X.shape

    def dist(X):
        """"""
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)

        np.fill_diagonal(D, 0)

        return D

    def entropy():
        """"""
        return np.log2(perplexity)

    D = dist(X)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = entropy()

    return D, P, betas, H
