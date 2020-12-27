#!/usr/bin/env python3
""""""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """"""
    res = []
    d_vars = []
    var = 0
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        if k == kmin:
            new_var = var
        res.append((C, clss))
        if isinstance(var, float):
            d_vars.append(new_var - var)
    
    return res, d_vars

