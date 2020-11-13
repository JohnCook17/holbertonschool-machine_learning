#!/usr/bin/env python3
"""The back ward step of a hmm"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Goes backwards, given the information"""
    # a = Transition (5, 5)
    # b = Emission shape = (5, 6)
    # V = Observation shape = (365,)

    T = Observation.shape[0]
    N = Transition.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))

    for t in range(T - 2, -1, -1):
        for j in range(N):
            beta[j, t] = ((beta[:, t + 1] *
                           Emission[:, Observation[t + 1]]).
                          dot(Transition[j, :]))

    likelihood = np.sum(np.sum(Initial.T * Emission[:, Observation[0]]
                               * beta[:, 0]))

    return likelihood, beta
