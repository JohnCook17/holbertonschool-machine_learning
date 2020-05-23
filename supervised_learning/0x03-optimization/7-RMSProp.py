#!/usr/bin/env python3
"""Root mean square or RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """alpha is the learning rate. beta2 is the RMSProp weight.
    epsilon is a small number to avoid zero division.
    var is either a number or a np.ndarray of numbers.
    grad is either a number or np.ndarrray of numbers.
    s is the previous second moment of var"""
    try:
        St = beta2 * s + (1 - beta2) * (grad ** 2)
        var = var - alpha * (grad / ((St ** .5) + epsilon))
        s = St
        return var, s
    except TypeError:
        for variable, gradient in zip(var, grad):
            St = beta2 * s + (1 - beta2) * (gradient ** 2)
            variable = variable - alpaha * (gradient / ((St ** .5) + epsilon))
            s = St
        return variable, s
