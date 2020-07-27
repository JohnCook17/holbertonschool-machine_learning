#!/usr/bin/env python3
"""Correlation matrix"""
import numpy as np


def correlation(C):
    """Returns the correlation matrix of a square matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) > 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    return np.corrcoef(C)
