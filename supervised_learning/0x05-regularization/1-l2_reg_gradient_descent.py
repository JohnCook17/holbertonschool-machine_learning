#!/usr/bin/env python3
"""performes l2 regularization gradeint descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Y is the one hot labels. weights is a dict. with weights and biases
    cache contains the output of each layer. lambtha is a traning perameter.
    L is the number of layers."""
    m = Y.shape[1]
    new_weights = weights
    for layer in range(L, 0, -1):
        if layer == L:
            # last layer is softmax
            dz = cache["A" + str(layer)] - Y
        else:
            # other layers are tanh
            dz = da * (1 - cache["A" + str(layer)] ** 2)
        l2_w = (lambtha / (2 * m)) * new_weights["W" + str(layer)]
        dw = np.matmul(dz, cache["A" + str(layer - 1)].T) + l2_w
        db = np.sum(dz, axis=1, keepdims=True)
        da = np.matmul(new_weights["W" + str(layer)].T, dz)
        # set new weights
        new_weights["W" + str(layer)] = (new_weights["W" + str(layer)] -
                                         alpha * dw)
        new_weights["b" + str(layer)] = (new_weights["b" + str(layer)] -
                                         alpha * db)
    weights = new_weights
