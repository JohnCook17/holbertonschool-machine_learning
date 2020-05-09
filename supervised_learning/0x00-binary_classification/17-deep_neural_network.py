#!/usr/bin/env python3
"""A deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """The deep neural network class"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        tmp_weights = {}
        for value in range(0, len(layers)):
            if not isinstance(layers[value], int):
                raise TypeError("layers must be a list of positive integers")
            if layers[value] < 1:
                raise TypeError("layers must be a list of positive integers")
            if value == 0:
                Wkey = "W" + str(value + 1)
                bkey = "b" + str(value + 1)
                tmp_weights[Wkey] = (np.random.randn(layers[value], nx) *
                                     np.sqrt(2 / nx))
                tmp_weights[bkey] = np.zeros((layers[value], 1))
            else:
                p = value - 1
                Wkey = "W" + str(value + 1)
                bkey = "b" + str(value + 1)
                tmp_weights[Wkey] = (np.random.randn(layers[value],
                                                     layers[(p)]) *
                                     np.sqrt(2 / layers[(p)]))
                tmp_weights[bkey] = np.zeros((layers[value], 1))
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = tmp_weights

    @property
    def L(self):
        """Getter for L, length of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache, the cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights, a dict containing the weights and biases"""
        return self.__weights
