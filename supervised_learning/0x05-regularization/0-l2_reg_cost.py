#!/usr/bin/env python3
"""l2 forward propagation"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """cost is the cost of the neural network, lambtha is trainable perameter
    L is the number of layers of the network, and m is the number of inputs"""
    new_weights = []
    for layer in range(1, L + 1):
        weight = weights["W" + str(layer)]
        weight = np.linalg.norm(weight, ord="fro")
        new_weights.append(weight)
    new_weights = np.asarray(new_weights)
    return np.sum(cost, keepdims=True) + ((lambtha / (2 * m)) *
                                          np.sum(new_weights, keepdims=True))
