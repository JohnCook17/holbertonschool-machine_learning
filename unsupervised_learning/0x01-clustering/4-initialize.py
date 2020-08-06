#!/usr/bin/env python3
""""""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """"""
    n, d = X.shape
    pi = np.ones(k) / k
    m = kmeans(X, k)[0]
    S = np.repeat(np.identity(d)[None, :], k, axis=0)
    return pi, m, S
