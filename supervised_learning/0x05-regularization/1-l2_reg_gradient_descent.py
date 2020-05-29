#!/usr/bin/env python3
""""""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """"""
    m = Y.shape[1]
    new_weights = {}
    # print(L)
    for layer in (L, 0, -1):
        print(layer)
        if layer == L:
            # last layer is softmax
            dz = cache["A" + str(layer)] - Y
        else:
            # other layers are tanh
            dz = 1 - cache["A" + str(layer)] ** 2
        # print(dz.shape)
        l2_w = (lambtha / m) * weights["W" + str(layer)]
        l2_b = (lambtha / m) * weights["b" + str(layer)]
        # print(l2_w.shape)
        # print(cache["A" + str(layer)].shape)
        dw = np.matmul(dz, cache["A" + str(layer)].T) + l2_w
        db = np.matmul(dz, cache["A" + str(layer)].T) + l2_b
        # set new weights
        new_weights["W" + str(layer)] = weights["W" + str(layer)] - alpha * dw
        new_weights["b" + str(layer)] = weights["b" + str(layer)] - alpha * db
    weights = new_weights
