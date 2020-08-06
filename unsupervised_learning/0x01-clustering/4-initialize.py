#!/usr/bin/env python3
""""""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """"""
    try:
        if k < 1:
            return None, None, None
        n, d = X.shape
        pi = np.ones(k) / k
        m = kmeans(X, k)[0]
        S = np.repeat(np.identity(d)[None, :], k, axis=0)
        return pi, m, S
    except Exception as e:
        return None, None, None
