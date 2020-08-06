#!/usr/bin/env python3
""""""
import numpy as np


def pdf(X, m, S):
    """"""
    n, d = X.shape
    det = np.linalg.det(S)
    first_term = (1 / ((2 * np.pi) ** (d / 2) * (det ** 0.5)))
    Xm = X - m
    X_t = ((np.linalg.inv(S)) @ Xm.T).T
    second_term = np.exp(-.5 * np.sum(Xm * X_t, axis=1))
    answer = first_term * second_term.T
    return np.where(answer == 0, 1e-300, answer)
