#!/usr/bin/env python3
"""The Viterbi algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """The viterbi algorithm, looks forward and back"""
    # a = Transition (5, 5)
    # b = Emission shape = (5, 6)
    # V = Observation shape = (365,)

    T = Observation.shape[0]
    N = Transition.shape[0]
    M = Transition.shape[1]

    omega = np.zeros((T, M))
    omega[0, :, np.newaxis] = (Emission[:, Observation[0]] * Initial.T).T

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            prob = np.multiply(omega[t - 1], Transition[:, j])
            omega[t, j] = np.max(prob) * Emission[j, Observation[t]]
            prev[t - 1, j] = np.argmax(prob)

    S = np.zeros(T)

    last_state = np.argmax(omega[T - 1, :])

    S[-1] = last_state

    back_indx = 1
    for i in range(T - 2, -1, -1):
        S[i] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]

    return S, np.max(omega[T - 1, :])
