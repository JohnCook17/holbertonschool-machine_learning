#!/usr/bin/env python3
"""E step of gmm"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Performs the 'e' step of gmm"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None

    d = X.shape[1]
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None

    if S.shape[0] != K or S.shape[1] != d or S.shape[2] != d:
        return None, None

    numerator = []
    for i in range(k):
        numerator.append(pi[i] * pdf(X, m[i], S[i]))
    numerator = np.asarray(numerator)

    likelyhood = np.sum(np.log(np.sum(numerator, axis=0)))

    g = numerator / np.sum(numerator, axis=0)

    return g, likelyhood
