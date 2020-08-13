#!/usr/bin/env python3
"""Determines if a Markov chain is abosorbing"""
import numpy as np


def absorbing(P):
    """P is the Markov chain to test"""
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    try:
        # print(P)
        # print("===================================")
        i = 2
        while i < 10:
            # print(np.power(P, i))
            # print(np.count_nonzero(P, axis=0))
            if ((np.any(np.isclose(np.power(P, i), 1)) and
                 np.all(np.count_nonzero(P, axis=0) > 1))):
                return True
            elif np.all(np.diag(P) == 1):
                return True
            i += 1
        return False
    except Exception as e:
        # print(e)
        return False
