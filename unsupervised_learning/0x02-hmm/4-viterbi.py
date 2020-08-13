#!/usr/bin/env python3
"""Viterbi calculation for HMM"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """np implementation of the Viterbi formula"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    V = np.zeros((N, T))
    B = np.zeros((N, T))
    for s in range(N):
        V[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            temp = V[:, t - 1] * Transition[:, s] * Emission[s, Observation[t]]
            V[s, t] = max(temp)
            B[s, t] = np.argmax(temp)
    prob = max(V[:, T - 1])
    S = np.argmax(V[:, T - 1])
    path = [S]
    for t in range(T - 1, 0, -1):
        S = int(B[S, t])
        path.append(S)
    return path[::-1], prob
