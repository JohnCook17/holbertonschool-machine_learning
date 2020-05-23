#!/usr/bin/env python3
"""Adaptive Momentum or ADAM a combination of RSMProp and Momentum"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """alpha is the learning rate. beta1 is the weight for the first moment.
    beta2 is the weight for the second moment. epsilon avoids division by zero.
    var is either a number or np.ndarray of numbers. grad is either a number or
    np.ndarray of numbers. v is the previous first moment. s is the previous
    second moment. t is the time step used for bias correction."""
    try:
        Vt = beta1 * v + (1 - beta1) * grad
        St = beta2 * s + (1 - beta2) * grad ** 2
        v = Vt
        s = St
        Vt = Vt / (1 - beta1 ** t)
        St = St / (1 - beta2 ** t)
        var = var - alpha * (Vt / ((St ** .5) + epsilon))
        return var, v, s
    except TypeError:
        for variable, gradient in zip(var, grad):
            Vt = beta1 * v + (1 - beta1) * grad
            St = beta2 * s + (1 - beta2) * grad ** 2
            v = Vt
            s = St
            Vt = Vt / (1 - beta1 ** t)
            St = St / (1 - beta2 ** t)
            variable = variable - alpha * (Vt / ((St ** .5) + epsilon))
        return variable, v, s
