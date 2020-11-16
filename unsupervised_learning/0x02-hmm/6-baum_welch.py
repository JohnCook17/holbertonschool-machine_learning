#!/usr/bin/env python3
"""The Baum-Welch Algorithm"""
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """An implementation of the Baum-Welch algorithm"""
    # a = Transition (5, 5)
    # b = Emission shape = (5, 6)
    # V = Observation shape = (365,)

    M = Transition.shape[0]
    T = Observations.shape[0]

    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)[1].T
        beta = backward(Observations, Emission, Transition, Initial)[1].T

        old_T = Transition
        old_E = Emission

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, Transition) *
                                 Emission[:, Observations[t + 1]].T,
                                 beta[t + 1, :])
            for i in range(M):
                numerator = (alpha[t, i] * Transition[i, :] *
                             Emission[:, Observations[t + 1]].T *
                             beta[t + 1, :])
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

        if ((np.isclose(old_T, Transition).any() or
             np.isclose(old_E, Emission).any())):
            return Transition, Emission

    return Transition, Emission
