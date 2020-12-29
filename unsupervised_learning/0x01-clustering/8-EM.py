#!/usr/bin/env python3
""""""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k < 1:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    previous_l = 0

    for i in range(iterations):
        g, likelyhood = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:0.5f}".format(i, likelyhood))

        pi, m, S = maximization(X, g)

        if np.isclose(likelyhood, previous_l, atol=tol, rtol=0):
            break

        previous_l = likelyhood

    if verbose:
        print("Log Likelihood after {} iterations: {:0.5f}".format(i, likelyhood))

    return pi, m, S, g, likelyhood
