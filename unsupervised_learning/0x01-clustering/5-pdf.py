#!/usr/bin/env python3
""""""
import numpy as np


def pdf(X, m, S):
    """"""
    n, d = X.shape
    first_term = (1 / ((2 * np.pi) ** (d / 2) * ((np.linalg.norm(S, axis=0, keepdims=True) ** 0.5))))
    print(first_term.shape)
    X_cent = X - m
    print(X_cent.shape)
    X_t = X_cent @ (np.linalg.inv(S))
    print(X_t.shape)
    second_term = np.exp(-.5 * X_t * (X_cent))
    print(second_term.shape)
    answer = first_term @ second_term.T
    print(answer.shape)
    return answer.flatten()
