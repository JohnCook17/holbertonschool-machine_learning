#!/usr/bin/env python3
"""helps find the optimum K"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """finds the optimum k"""
    try:
        if not isinstance(kmin, int) or not isinstance(kmax, int):
            return None
        if kmin < 1:
            return None
        if kmin >= kmax:
            return None
        res = []
        var_list = []
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            res.append((C, clss))
            var = variance(X, C)
            if k == kmin:
                var_first = var
            var_list.append(var_first - var)
        return res, var_list
    except Exception as e:
        return None, None
