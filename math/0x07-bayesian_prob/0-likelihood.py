#!/usr/bin/env python3
""""""
import numpy as np


def likelihood(x, n, P):
    """P(A | B) = P(B | A) * P(A) / P(B)"""
    if n < 1:
        raise ValueError("n must be a positive integer")
    if x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P) < 0 or np.any(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    fac = np.math.factorial
    return (fac(n) / (fac(x) * fac(n - x))) * (P ** x) * ((1 - P) ** (n - x))
