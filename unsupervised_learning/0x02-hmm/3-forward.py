#!/usr/bin/env python3
"""Forward step in hmm"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Forward step of hmm"""
    N = Emission.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    for s in range(N):
        F[s, 0] = Initial[s, 0] * Emission[s, Observation[0]]
    for t, o in enumerate(Observation):
        if t == 0:
            continue
        for s in range(N):
            F[s, t] = np.sum(F[s, t - 1] * Transition[:, s]) * Emission[s, o]
    P = np.sum(F[:, T - 1])
    return P, F
