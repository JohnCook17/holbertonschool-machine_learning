#!/usr/bin/env python3
"""A binary classification neural network"""
import numpy as np


class NeuralNetwork:
    """The NeuralNetwork class for binary classification"""
    def __init__(self, nx, nodes):
        """init of weights, biases, and activated output based on nx,
        the number of input features, and nodes the number of nodes
        found in the layer."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
