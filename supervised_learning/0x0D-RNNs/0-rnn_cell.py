#!/usr/bin/env python3
""""""
import numpy as np


class RNNCell():
    """"""
    def __init__(self, i, h, o):
        """"""
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def softmax(self, x):
        """"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        term0 = np.matmul(concat, self.Wh)
        h_next = np.tanh(term0 + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
