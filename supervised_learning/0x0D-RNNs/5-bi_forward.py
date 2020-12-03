#!/usr/bin/env python3
""""""
import numpy as np


class BidirectionalCell():
    """"""
    def __init__(self, i, h, o):
        """"""
        self.Whf = np.random.normal(size=(h, h + i))
        self.Whb = np.random.normal(size=(h, h + i))
        self.Wy = np.random.normal(size=(o, h))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, h_prev, x_t):
        """"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        term0 = np.matmul(concat, self.Wh)
        h_next = np.tanh(term0 + self.bh)

        return h_next
