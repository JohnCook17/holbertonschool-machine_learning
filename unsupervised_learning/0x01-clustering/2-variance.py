#!/usr/bin/env python3
"""Intra cluster variance"""
import numpy as np


def variance(X, C):
    """Finds the variance of X clusters with C centers"""
    if not isinstance(C, np.ndarray):
        return None

    if len(C.shape) != 2:
        return None

    if not isinstance(X, np.ndarray):
        return None

    if C.shape[1] != X.shape[1]:
        return None

    if C.shape[0] < 1:
        return None

    return (np.sum(np.sum(np.min(np.linalg.norm(X[:, :, np.newaxis] -
                                                C[:, :, np.newaxis]
                                                .T, axis=1),
                                 axis=1) ** 2)))
