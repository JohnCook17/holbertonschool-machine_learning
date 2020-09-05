#!/usr/bin/env python3
""""""
import numpy as np


class RNNCell():
    """A cell of a simple RNN"""
    def __init__(self, i, h, o):
        """init of cell, i is the dim of the data
        h is the dim of the hidden state
        o is the dim of the output"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """performs the forward prop for one time step"""
        def softmax(x):
            """softmax function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        x_prev = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(x_prev, self.Wh) + self.bh)
        y = softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
