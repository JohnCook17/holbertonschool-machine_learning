#!/usr/bin/env python3
"""Checks if a Markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """Uses a series of conditional statements to determine if absorbing"""
    if np.all(np.diag(P) == 1):
        return True

    if P[0, 0] != 1:
        return False

    P = P[1:, 1:]

    if np.all(np.count_nonzero(P, axis=0) > 2):
        return True
    else:
        return False
