#!/usr/bin/env python3
"""Gradient descent with momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """alpha is the learning rate. beta1 is the momentum weight.
    var is either a number of list of numbers in a np.ndarray.
    grad is either a number or a list of numbers in a np.ndarray.
    v is the previous moment of the first var."""
    try:
        Vt = beta1 * v + (1 - beta1) * grad
        v = Vt
        var = var - (alpha * Vt)
        return var, v
    except TypeError:
        for variable, gradient in zip(var, grad):
            Vt = beta1 * v + (1 - beta1) * gradient
            v = Vt
            variable = variable - (alpha * Vt)
        return variable, v
