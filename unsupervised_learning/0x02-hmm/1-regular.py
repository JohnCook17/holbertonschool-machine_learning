#!/usr/bin/env python3
"""determines the steady state of a Markov chain"""
import numpy as np


def regular(P):
    """P is the matrix to find the steady state of"""
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    try:
        d = P.shape[0]
        q = (P - np.eye(d))
        ones = np.ones(d)
        q = np.c_[q, ones]
        QT = np.dot(q, q.T)
        res = np.linalg.solve(QT, ones)
        if np.any(res < 0):
            return None
        return res
    except Exception as e:
        return None
