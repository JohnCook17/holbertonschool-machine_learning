#!/usr/bin/env python3
""""""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """"""
    pi = np.array(np.repeat((1 / k), k))
    m = kmeans(X, k)
    identity_m = np.identity(X.shape[1])
    print(identity_m)
    S = np.repeat(identity_m[np.newaxis, ...], k, axis=0)

    return pi, m, S
