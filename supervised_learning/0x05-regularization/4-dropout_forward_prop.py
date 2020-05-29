#!/usr/bin/env python3
"""Drop out reg"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward prop using drop out"""
    cache = {}
    nx = X.shape[0]
    m = X.shape[1]
    cache["A0"] = X
    for layer in range(1, L + 1):
        Z = (np.matmul(weights["W" + str(layer)], cache["A" + str(layer - 1)])
             + weights["b" + str(layer)])
        if layer == L:
            e_x = np.exp(Z - np.max(Z))
            cache["A" + str(layer)] = e_x / np.sum(e_x, axis=0)
        else:
            cache["A" + str(layer)] = np.tanh(Z)
            cache["D" + str(layer)] = (np.where
                                       (np.random.rand
                                        (cache["A" + str(layer)].shape[0],
                                         cache["A" + str(layer)].shape[1]) <
                                        keep_prob, 1, 0))
            cache["A" + str(layer)] *= cache["D" + str(layer)]
            cache["A" + str(layer)] /= keep_prob
    return cache
