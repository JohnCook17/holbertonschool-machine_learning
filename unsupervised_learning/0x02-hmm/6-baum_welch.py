#!/usr/bin/env python3
"""Baum Welch algorithm"""
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
    """Uses forward and backward algos above to calculate new values for
    Transition and Emission, note to self, watch this video if stuck:
    https://www.youtube.com/
    watch?v=JRsdt05pMoI&list=PLix7MmR3doRo3NGNzrq48FItR3TDyuLCo&index=12"""
    M = Transition.shape[0]
    T = Observations.shape[0]
    a = Transition
    b = Emission
    V = Observations

    for n in range(iterations):
        alpha = forward(V, b, a, Initial)[1].T
        beta = backward(V, b, a, Initial)[1].T
        a_prev = a
        b_prev = b

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.matmul(np.matmul(alpha[t, :].T, a) *
                                    b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = (alpha[t, i] * a[i, :] * b[:, V[t + 1]].T *
                             beta[t + 1, :].T)
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

        if ((np.isclose(a_prev, a, atol=1e-5).all() or
             np.isclose(b_prev, b, atol=1e-5).all())):
            # print("early stop")
            return a, b
        # else:
            # print(n)
    return a, b
