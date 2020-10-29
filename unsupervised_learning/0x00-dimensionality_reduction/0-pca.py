#!/usr/bin/env python3
"""PCA of an array to reduce the number of features"""
import numpy as np


def pca(X, var=0.95):
    """performs pca on a matrix"""
    W, V = np.linalg.eig(np.matmul(X.T, X))
    W_idx = W.argsort()[::-1]
    V = V[:, W_idx]
    # print(V)
    V_var = np.copy(V)
    V_var *= 1 / np.abs(V_var).max()
    # print(V_var)
    V_idx = V[np.where(np.abs(V_var) >= var, True, False)]
    # print(V_idx.shape)
    V_idx = len(V_idx)
    # print(V[:, :V_idx].shape)
    return V[:, :V_idx] * -1.
