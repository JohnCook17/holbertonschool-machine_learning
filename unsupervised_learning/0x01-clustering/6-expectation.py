#!/usr/bin/env python3
""""""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """"""
    k = pi.shape[0]
    g = []
    for i in range(k):
        N = pdf(X, m[i], S[i])
        numerator = pi[i] * N
        # print(numerator.shape)
        g.append(numerator)
    g = np.asarray(g)
    likelihood = np.sum(np.log(np.matmul(pi[np.newaxis], g) * k))
    # g /= np.sum(pi) * np.sum(g, axis=1)
    g /= np.matmul(pi[np.newaxis], g) * k
    return g, likelihood
