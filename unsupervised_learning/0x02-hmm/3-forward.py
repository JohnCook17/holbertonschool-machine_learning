#!/usr/bin/env python3
"""The forward step of hmm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """The forward step of hmm"""
    # a = Transition (5, 5)
    # b = Emission shape = (5, 6)
    # V = Observation shape = (365,)
    T = Observation.shape[0]
    N = Transition.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0, np.newaxis] = (Initial.T * Emission[:, Observation[0]]).T

    for t in range(1, Observation.shape[0]):
        for j in range(Transition.shape[0]):
            alpha[j, t] = (alpha[:, t - 1].dot(Transition[:, j]) *
                           Emission[j, Observation[t]])

    prob = np.sum(alpha[:, -1])

    return prob, alpha
