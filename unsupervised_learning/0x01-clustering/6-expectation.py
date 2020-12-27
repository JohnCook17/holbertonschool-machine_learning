#!/usr/bin/env python3
"""E step of gmm"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Performs the 'e' step of gmm"""
    k = pi.shape[0]
    numerator = []
    for i in range(k):
        numerator.append(pi[i] * pdf(X, m[i], S[i]))
    numerator = np.asarray(numerator)

    l = np.sum(np.log(np.sum(numerator, axis=0)))
    
    g = numerator / np.sum(numerator, axis=0)

    return g, l
