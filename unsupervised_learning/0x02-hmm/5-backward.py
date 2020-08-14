#!/usr/bin/env python3
"""A numpy version of backward"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Backward algorithm for HMM"""
    N = Emission.shape[0]
    T = Observation.shape[0]
    B = np.zeros((N, T))
    for s in range(N - 1, -1, -1):
        B[s, T - 1] = 1
    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(Transition[s, :] * Emission[:, Observation[t + 1]]
                             * B[:, t + 1])
    P = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
    return P, B
