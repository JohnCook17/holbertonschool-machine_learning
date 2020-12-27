#!/usr/bin/env python3
"""Finds the optimal number of clusters"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Provides info for optimal cluster number"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = iterations

    if kmin is None:
        kmin = 1

    if kmin < 1:
        return None, None

    if kmax <= kmin:
        return None, None
    
    res = []
    d_vars = []
    var = 0
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        if k == kmin:
            new_var = var
        if C is not None and Clss is not None:
            res.append((C, clss))
        if isinstance(var, float):
            d_vars.append(new_var - var)
    
    return res, d_vars

