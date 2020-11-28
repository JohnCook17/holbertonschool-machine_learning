#!/usr/bin/env python3
"""k-means clustering"""
import numpy as np


def initialize(X, k):
    """Initializes for K means given the data X"""
    if k < 1:
        return None
    try:
        d = X.shape[1]
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        return np.random.uniform(low=data_min, high=data_max, size=(k, d))
    except Exception as e:
        return None
