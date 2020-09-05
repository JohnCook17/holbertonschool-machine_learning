#!/usr/bin/env python3
"""Gated Recurant Cell"""
import numpy as np


class GRUCell():
    """This Recurant Cell has a forget gate"""
    def __init__(self, i, h, o):
        """init the values needed"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward pass of gru cell"""
        def sigmoid(x):
            """sigmoid function"""
            return 1 / (1 + np.exp(-x))

        def softmax(x):
            """softmax function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        x_prev = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(np.matmul(x_prev, self.Wz) + self.bz)
        r = sigmoid(np.matmul(x_prev, self.Wr) + self.br)
        h_r = np.concatenate((h_prev * r, x_t), axis=1)
        g = np.tanh(np.matmul(h_r, self.Wh) + self.bh)
        h = (1 - z) * h_prev + z * g
        y = softmax((np.matmul(h, self.Wy) + self.by))
        return h, y
