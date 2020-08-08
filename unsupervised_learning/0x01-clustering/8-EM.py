#!/usr/bin/env python3
""""""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """"""
    i = 0
    log_likelihood = 0
    diff = 0
    pi, m, S = initialize(X, k)
    while i < iterations or not np.isclose(diff, np.round(log_likelihood, 5),
                                           rtol=tol):
        g, log_likelihood = expectation(X, pi, m, S)
        if i % 10 == 0 and verbose is True:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, np.round(log_likelihood, 5)))
        diff = np.round(log_likelihood, 5)
        pi, m, S = maximization(X, g)
        i += 1
    if verbose is True:
        print("Log Likelihood after {} iterations: {}"
              .format(i, np.round(log_likelihood, 5)))
    return pi, m, S, g, log_likelihood
