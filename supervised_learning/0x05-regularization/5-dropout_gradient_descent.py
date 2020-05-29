#!/usr/bin/env python3
""""""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """"""
    m = Y.shape[1]
    new_weights = weights
    for layer in range(L, 0, -1):
        if layer == L:
            # last layer is softmax
            dz = cache["A" + str(layer)] - Y
        else:
            # other layers are tanh
            dz = da * (1 - cache["A" + str(layer)] ** 2)
            dz *= cache["D" + str(layer)]
            dz /= keep_prob
        dw = np.matmul(dz, cache["A" + str(layer - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(new_weights["W" + str(layer)].T, dz)
        # set new weights
        new_weights["W" + str(layer)] = (new_weights["W" + str(layer)] -
                                         alpha * dw)
        new_weights["b" + str(layer)] = (new_weights["b" + str(layer)] -
                                         alpha * db)
    weights = new_weights.copy()
