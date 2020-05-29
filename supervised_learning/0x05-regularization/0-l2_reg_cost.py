#!/usr/bin/env python3
""""""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """"""
    new_weights = []
    for layer in range(1, L + 1):
        weight = weights["W" + str(layer)]
        weight = np.linalg.norm(weight, ord="fro")
        new_weights.append(weight)
    new_weights = np.asarray(new_weights)
    return np.sum(cost, keepdims=True) + ((lambtha / (2 * m)) * np.sum(new_weights, keepdims=True))
