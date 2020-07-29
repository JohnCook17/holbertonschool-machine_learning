#!/usr/bin/env python3
""""""
import numpy as np


def pca(X, var=0.95):
    """"""
    # print(X.shape)
    u, s, vh = np.linalg.svd(X)
    # print(s.shape)
    keep_per = 0
    i = 0
    while keep_per < var:
        keep_per = sum(s[0:i]) / sum(s)
        # print(keep_per)
        if keep_per < var:
            i += 1
    w = vh.T
    return w[0:, 0:i]
