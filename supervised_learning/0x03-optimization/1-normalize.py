#!/usr/bin/env python3
"""Normalizes a matrix"""
import numpy as np


def normalize(X, m, s):
    """X is a np.ndarray with shape (d, nx) d is the number of data points
    nx is the number of features. m is a np.ndarray with shape (nx,) that
    contains the mean of all features of X. s is a np.ndarray with the shape
    of (nx,) that contains the standard deviation of all features of X.
    returns the normalized X matrix"""
    return (X - m) / s
