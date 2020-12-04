#!/usr/bin/env python3
""""""
import numpy as np


class BidirectionalCell():
    """"""
    def __init__(self, i, h, o):
        """"""
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        term0 = np.matmul(concat, self.Whf)
        h_next = np.tanh(term0 + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """"""
        concat = np.concatenate((h_next, x_t), axis=1)
        term0 = np.matmul(concat, self.Whb)
        h_prev = np.tanh(term0 +self.bhb)

        return h_prev
