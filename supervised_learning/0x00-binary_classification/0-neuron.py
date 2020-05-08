#!/usr/bin/env python3
"""A Neuron for binary classification"""
import numpy as np


class Neuron:
    """A Neuron class"""
    def __init__(self, nx):
        """inits nx the number of input features"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
