#!/usr/bin/env python3
"""marginal probabilities"""
import numpy as np


def marginal(x, n, P, Pr):
    """finds the marginal probabilities"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if ((np.any(np.where(P < 0, True, False)) is True or
         np.any(np.where(P > 1, True, False)) is True)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if ((np.any(np.where(P < 0, True, False)) is True or
         np.any(np.where(P > 1, True, False)) is True)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    fac = np.math.factorial
    likelyhood = ((fac(n) / (fac(x) * fac(n - x))) *
                  (P ** x) * ((1 - P) ** (n - x)))
    return np.sum(likelyhood * Pr)
