#!/usr/bin/env python3
"""Bidirectional Cell, forward pass"""
import numpy as np


class BidirectionalCell():
    """The Bidirectional Cell backwards pass"""
    def __init__(self, i, h, o):
        """The forward, backward and current weights and biases"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """The forward pass"""
        current = np.concatenate((h_prev, x_t), axis=1)
        f = np.tanh(np.matmul(current, self.Whf) + self.bhf)
        return f

    def backward(self, h_next, x_t):
        """The backward pass"""
        current = np.concatenate((h_next, x_t), axis=1)
        b = np.tanh(np.matmul(current, self.Whb) + self.bhb)
        return b
