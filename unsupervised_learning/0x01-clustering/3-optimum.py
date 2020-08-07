#!/usr/bin/env python3
"""helps find the optimum K"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """finds the optimum k"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = iterations
    if kmin is None:
        kmin = 1
    if not isinstance(kmin, int) or not isinstance(kmax, int):
        return None, None
    if kmin < 1:
        return None, None
    if kmin >= kmax:
        return None, None
    res = []
    var_list = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if isinstance(C, np.ndarray) and isinstance(clss, np.ndarray):
            res.append((C, clss))
        var = variance(X, C)
        if k == kmin:
            var_first = var
        if isinstance(var, float):
            var_list.append(var_first - var)
    return res, var_list
