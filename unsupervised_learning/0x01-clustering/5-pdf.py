#!/usr/bin/env python3
""""""
import numpy as np


def pdf(X, m, S):
    """"""
    tol = 1e-300
    i, j = np.indices(S.shape)
    variance = S[i == j]
    print(variance)
    term0 = 1 / (np.sqrt(2 * np.pi * variance))
    print("term0 = ", term0)
    term1 = np.exp(-(np.square(X - m) / (2 * variance)))
    print("term1 = ", term1)
    ret = term0 * term1
    print("ret = ", ret.shape)
    return np.where(ret < tol, tol, ret)
