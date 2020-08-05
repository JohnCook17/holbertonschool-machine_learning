#!/usr/bin/env python3
""""""
import numpy as np


def initialize(X, k):
    """takes an array X of shape (n, d) where n is the number of data points
    and d is the dimensions of each dp, and K the number of clusters"""
    try:
        if k < 1 or X.shape[1] < 1:
            return None
        x_dim = X.shape[1]
        x_max = np.max(X, axis=0)
        x_min = np.min(X, axis=0)
        return np.random.uniform(x_min, x_max, (k, x_dim))
    except Exception as e:
        return None


def class_assignment(X, C):
    """class assignment"""
    d = X[:, np.newaxis] - C
    diffs = np.sqrt(np.sum(d ** 2, axis=2))
    clss = np.argmin(diffs, axis=1)
    return clss


def move_centroids(X, C, clss):
    """moves centroids"""
    k = C.shape[1]
    for i in range(k):
        if X[clss == i].size == 0:
            x_max = np.max(X, axis=0)
            x_min = np.min(X, axis=0)
            C[i, :] = np.random.uniform(x_min, x_max, (1, k))
        else:
            C[i, :] = np.mean(X[clss == i], axis=0)
    return C


def kmeans(X, k, iterations=1000):
    """"""
    if not isinstance(X, np.ndarray):
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    n, d = X.shape
    if not isinstance(k, int) or k < 1 or k > n:
        return None, None
    C_prev_move = 0
    C = initialize(X, k)
    clss = []
    for i in range(iterations):
        # print(i)
        C_prev = np.copy(C)
        clss = class_assignment(X, C)
        C = move_centroids(X, C, clss)
        # print(C)
        C_move = np.linalg.norm(C - C_prev)
        if C_prev_move - C_move == 0:
            clss = class_assignment(X, C)
            C = move_centroids(X, C, clss)
            print(i)
            return C, clss
        C_prev_move = C_move
    clss = class_assignment(X, C)
    return C, clss
