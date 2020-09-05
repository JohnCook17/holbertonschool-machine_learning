#!/usr/bin/env python3
"""LSTM Cell - long short term memory cell"""
import numpy as np


class LSTMCell():
    """The LSTM Cell class"""
    def __init__(self, i, h, o):
        """init of the values needed"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """forward pass"""
        def sigmoid(x):
            """sigmoid function"""
            return 1 / (1 + np.exp(-x))

        def softmax(x):
            """softmax function"""
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        z = np.concatenate((h_prev, x_t), axis=1)
        f = sigmoid(np.matmul(z, self.Wf) + self.bf)
        u = sigmoid(np.matmul(z, self.Wu) + self.bu)
        c_bar = np.tanh(np.matmul(z, self.Wc) + self.bc)

        c = f * c_prev + u * c_bar
        o = sigmoid(np.matmul(z, self.Wo) + self.bo)
        h = o * np.tanh(c)
        y = softmax((np.matmul(h, self.Wy) + self.by))
        return h, c, y
