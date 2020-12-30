#!/usr/bin/env python3
"""placeholder file"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Bayesian information criterion, finds the 'best'
       number of clusters for the data"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None

    if not isinstance(iterations, int):
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    if kmax is None:
        kmax = iterations

    n = X.shape[0]
    prior_bic = 0
    likelyhoods = []
    bics = []
    best_k = kmax
    for k in range(kmin, kmax + 1):
        # print(k)
        pi, m, S, g, likelyhood = expectation_maximization(X,
                                                           k,
                                                           iterations,
                                                           tol,
                                                           verbose)

        bic = k * np.log(n) - 2 * likelyhood

        if np.isclose(bic, prior_bic) and best_k >= k:
            best_k = k - 1
            best_res = pi_previous, m_previous, S_previous

        pi_previous, m_previous, S_previous = pi, m, S

        likelyhoods.append(likelyhood)
        bics.append(bic)

        prior_bic = bic

    return best_k, best_res, np.asarray(likelyhoods)[:-1], np.asarray(bics)[:-1]
