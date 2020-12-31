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


    def softmax(X):
        """"""
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        np.fill_diagonal(e_x, 0)

        return e_x / e_x.sum(axis=1).reshape([-1, 1])


    def prob_mat(dist):
        """"""

        return softmax(dist)


    def calc_perplexity(X):
        """"""
        entropy = -np.sum(X * np.log2(X))

        perplexity = 2 ** entropy

        return entropy


    def shannon_entropy(D, perplexity):
        """"""
        return calc_perplexity(perplexity)


    D = dist(X)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = shannon_entropy(X, perplexity)

    return D, P, betas, H
