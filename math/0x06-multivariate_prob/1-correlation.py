#!/usr/bin/env python3
"""finds the correlation given C a covariance matrix"""
import numpy as np


def correlation(C):
    """C is already a covariance matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    R = np.zeros((C.shape))
    for i in range(C.shape[0]):
        for j in range(C.shape[0]):
            R[i][j] = C[i][j] / np.sqrt(C[i][i] * C[j][j])

    return R
