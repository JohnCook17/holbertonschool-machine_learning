#!/usr/bin/env python3
""""""
import numpy as np


def pca(X, ndim):
    """"""
    X -= np.mean(X, axis=0)
    W, V = np.linalg.eig((np.cov(X.T)))
    # idx = W.argsort()[::-1]
    # V = V[:, idx]
    T = np.real(np.matmul(X, V[:, :ndim]))
    return T * -1.
