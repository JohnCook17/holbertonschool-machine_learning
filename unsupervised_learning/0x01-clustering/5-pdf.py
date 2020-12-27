#!/usr/bin/env python3
"""PDF of gmm"""
import numpy as np


def pdf(X, m, S):
    """PDF of gaussian mixture model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    n, d = X.shape

    if not isinstance(m, np.ndarray) or len(m.shape) != 1 or m.shape[0] != d:
        return None

    if ((not isinstance(S, np.ndarray)
         or len(S.shape) != 2
         or S.shape[0] != d
         or S.shape[1] != d)):
        return None

    tol = 1e-300
    sigma = np.linalg.det(S)
    term0 = (1 / ((2 * np.pi) ** (d / 2) * (sigma ** 0.5)))
    term1 = (np.linalg.inv(S) @ (X - m).T).T
    term2 = np.exp(-0.5 * np.sum((X - m) * term1, axis=1))
    ret = term0 * term2.T
    return np.where(ret <= tol, tol, ret)
