#!/usr/bin/env python3
"""Variance in each cluster"""
import numpy as np


def variance(X, C):
    """Cluster variance, to find how many clusters to use"""
    diff = X[:, :, np.newaxis] - C[:, :, np.newaxis].T
    norm = np.linalg.norm(diff, axis=1)
    mini = np.min(norm, axis=1)
    sqr = mini ** 2
    sums = np.sum(np.sum(sqr))
    return sums
