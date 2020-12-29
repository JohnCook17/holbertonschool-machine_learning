#!/usr/bin/env python3
"""Performs 1 step of maximization"""
import numpy as np


def maximization(X, g):
    """Takes X the data and g the centroids"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    if np.isclose(np.sum(g), 1):
        return None, None, None

    k = g.shape[0]
    n = X.shape[0]

    N = []
    pi = []
    m = []
    S = []

    # print(g.shape)

    for j in range(k):
        N.append(np.sum(g[j]))

        pi.append(N[j] / n)

        m.append((g[j, np.newaxis] @ X / N[j]).flatten())

        S.append((g[j, np.newaxis] * (X - m[j]).T) @ (X - m[j]) / N[j])

    return np.asarray(pi), np.asarray(m), np.asarray(S)
