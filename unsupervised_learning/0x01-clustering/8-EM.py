#!/usr/bin/env python3
"""Expectation maximization"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs expectation maximization oh the X data set, with k clusters"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k < 2:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None
    i = 0
    log_likelihood = 0
    diff = 0
    pi, m, S = initialize(X, k)
    while i < iterations:
        g, log_likelihood = expectation(X, pi, m, S)
        if i % 10 == 0 and verbose:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, np.round(log_likelihood, 5)))
        pi, m, S = maximization(X, g)
        if np.isclose(diff, log_likelihood, atol=tol, rtol=0):
            break
        i += 1
        diff = log_likelihood
    if verbose:
        print("Log Likelihood after {} iterations: {}"
              .format(i, np.round(log_likelihood, 5)))
    return pi, m, S, g, log_likelihood
