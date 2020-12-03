#!/usr/bin/env python3
""""""
import numpy as np


class GRUCell():
    """"""
    def __init__(self, i, h, o):
        """"""
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros(shape=(1, h))
        self.br = np.zeros(shape=(1, h))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def sigmoid(self, x):
        """"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, h_prev, x_t):
        """"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z_t = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r_t = self.sigmoid(np.matmul(concat, self.Wr) + self.br)
        h_hat_t = np.tanh(np.matmul(concat, self.Wh) + (r_t * h_prev) + self.bh)
        h_t = (1 - z_t) * h_prev + z_t * h_hat_t
        y = self.softmax(np.matmul(h_t, self.Wy) + self.by)

        return h_t, y
