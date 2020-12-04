#!/usr/bin/env python3
"""LSTM Cell class"""
import numpy as np


class LSTMCell():
    """The LSTM Cell class"""
    def __init__(self, i, h, o):
        """init of LSTM"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros(shape=(1, h))
        self.bu = np.zeros(shape=(1, h))
        self.bc = np.zeros(shape=(1, h))
        self.bo = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def sigmoid(self, x):
        """The sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """The softmax function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """Forward prop of the LSTM"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        f_t = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u_t = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        o_t = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        c_hat_t = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_t = f_t * c_prev + u_t * c_hat_t
        h_t = o_t * np.tanh(c_t)

        y = self.softmax(np.matmul(h_t, self.Wy) + self.by)

        return h_t, c_t, y
