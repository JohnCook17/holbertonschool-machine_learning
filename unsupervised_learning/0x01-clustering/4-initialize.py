#!/usr/bin/env python3
"""Init gmm"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """uses kmeans to init gaussian mixture model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    pi = np.array(np.repeat((1 / k), k))
    m = kmeans(X, k)[0]
    identity_m = np.identity(X.shape[1])
    # print(identity_m)
    S = np.repeat(identity_m[np.newaxis, ...], k, axis=0)

    return pi, m, S
