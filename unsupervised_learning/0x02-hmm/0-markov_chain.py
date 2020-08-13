#!/usr/bin/env python3
"""A Markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """P is the transition matrix, s is the transition matrix,
    and t is the number of iterations"""
    if ((not isinstance(P, np.ndarray) or
         len(P.shape) != 2 or
         P.shape[0] != P.shape[1])):
        return None
    if ((not isinstance(s, np.ndarray) or
         len(s.shape) != 2 or
         s.shape[0] != 1 or
         s.shape[1] != P.shape[1])):
        return None
    if not isinstance(t, int) or t < 1:
        return None
    for i in range(t):
        s = np.matmul(s, P)
    return s
