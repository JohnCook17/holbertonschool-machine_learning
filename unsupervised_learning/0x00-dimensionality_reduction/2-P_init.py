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
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        np.fill_diagonal(e_x, 0.)

        e_x = e_x + 1e-8

        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    def calc_prob_mat(dist):
        two_sig_sq = 2. * np.square(perplexity)

        return softmax(dist / two_sig_sq)


    def entropy(prob_mat):
        """"""
        return np.log2(perplexity)

    D = dist(X)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    prob_mat = calc_prob_mat(D)

    H = entropy(prob_mat)

    return D, P, betas, H