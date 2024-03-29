#!/usr/bin/env python3
"""A Neuron for binary classification"""
import numpy as np


class Neuron:
    """A Neuron class"""
    def __init__(self, nx):
        """checks, nx the number of input features, inits Weights as W,
        Biases as b, and Activated Output as A"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """The weights getter"""
        return self.__W

    @property
    def b(self):
        """The bias getter"""
        return self.__b

    @property
    def A(self):
        return self.__A
