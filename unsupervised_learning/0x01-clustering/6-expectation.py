#!/usr/bin/env python3
"""Expectation step"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """performes expectation returns probs and likelihood"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if ((not isinstance(pi, np.ndarray) or
         len(pi.shape) != 1 or
         not np.isclose(1, np.sum(pi)))):
        return None, None
    if ((not isinstance(m, np.ndarray) or
         len(m.shape) != 2 or
         m.shape[1] != X.shape[1] or
         m.shape[0] != pi.shape[0])):
        return None, None
    if ((not isinstance(S, np.ndarray) or
         len(S.shape) != 3 or
         S.shape[0] != pi.shape[0] or
         S.shape[1] != X.shape[1] or
         S.shape[2] != X.shape[1])):
        return None, None
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
