#!/usr/bin/env python3
"""A policy to follow"""
import numpy as np


def policy(matrix, weights):
    """The policy we will use"""
    # for each col of weights we sum wi*si
    z = matrix.dot(weights)
    # for same results
    exp = np.exp(z)
    return exp / np.sum(exp)



def policy_gradient(state, weight):
    """the policy gradient"""
    P = policy(state, weight)
    action = np.random.choice(len(P[0]), p=P[0])

    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)

    dsoftmax = softmax[action, :]

    dlog = dsoftmax / P[0, action]
    grad = state.T.dot(dlog[None, :])

    return action, grad
