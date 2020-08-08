#!/usr/bin/env python3
"""performes the m-step of gmm"""
import numpy as np


def maximization(X, g):
    """Performes the maximization step"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    n, d = X.shape
    K, _ = g.shape
    N = []
    pi = []
    m = []
    S = []
    for k in range(K):
        N.append(np.sum(g[k]))
        pi.append(N[k] / n)
        m.append((g[k][np.newaxis] @ X / N[k]).flatten())
        Xm = X - m[k]
        S.append((g[k] * Xm.T) @ Xm / N[k])
    pi = np.asarray(pi)
    m = np.asarray(m)
    S = np.asarray(S)
    return pi, m, S
