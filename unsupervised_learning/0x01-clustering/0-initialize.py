#!/usr/bin/env python3
"""initializes K-means centroid"""
import numpy as np


def initialize(X, k):
    """takes an array X of shape (n, d) where n is the number of data points
    and d is the dimensions of each dp, and K the number of clusters"""
    try:
        if k < 1 or X.shape[1] < 1:
            return None
        x_dim = X.shape[1]
        x_max = np.max(X, axis=0)
        x_min = np.min(X, axis=0)
        return np.random.uniform(x_min, x_max, (k, x_dim))
    except Exception as e:
        return None
