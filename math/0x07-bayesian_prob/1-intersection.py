#!/usr/bin/env python3
""""""
import numpy as np


def likelihood(x, n, P):
    """"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) and x < 0:
        raise ValueError("x must be an integer that is",
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    likelihoods = ((np.math.factorial(n)) / (np.math.factorial(x) *
                   (np.math.factorial(n - x))) * P ** x * (1 - P) ** (n - x))

    return likelihoods


def intersection(x, n, P, Pr):
    """"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) and x < 0:
        raise ValueError("x must be an integer that is",
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")
    A = likelihood(x, n, P)
    intersect = A * Pr

    return intersect
