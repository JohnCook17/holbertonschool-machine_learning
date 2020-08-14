#!/usr/bin/env python3
""""""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Forward step of hmm"""
    N = Emission.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    for s in range(N):
        F[s, 0] = Initial[s, 0] * Emission[s, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.sum(F[:, t - 1] * Transition[:, s] *
                             Emission[s, Observation[t]])
    P = np.sum(F[:, T - 1])
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """Backward algorithm for HMM"""
    N = Emission.shape[0]
    T = Observation.shape[0]
    B = np.zeros((N, T))
    for s in range(N - 1, -1, -1):
        B[s, T - 1] = 1
    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :]
                             * Emission[:, Observation[t + 1]])
    P = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
    return P, B


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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """"""
    M = Transition.shape[0]
    T = Observations.shape[0]
    for n in range(iterations):
        alpha = forward(Observations, Emission, Transition, Initial)[1].T
        beta = backward(Observations, Emission, Transition, Initial)[1].T
        xi = np.zeros((M, M, T))
        for t in range(T - 1):
            print(xi)
            denominator = np.dot(np.dot(alpha[t, :].T, Initial) * Emission[:, Observations[t + 1]].T,  beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * Initial[i, :] * Emission[:, Observations[t + 1]].T * beta[t + 1, :].T
                print(numerator, denominator)
                xi[i, :, t] = numerator / denominator
        gama = np.sum(xi, axis=1)
        # print(gama.shape)
        Transition = np.sum(xi, 2) / np.sum(gama, axis=1).reshape((-1, 1))
        gama = np.hstack((gama, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        gama = np.delete(gama, (0), axis=1)
        # print(gama.shape)
        K = Emission.shape[0]
        denominator = np.sum(gama, axis=1)
        for l in range(K):
            Emission[:, l] = np.sum(gama[:, Observations == l], axis=1)
        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
