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

    def forward_prop(self, X):
        """Forward propagation, with X as an np array with shape nx, m"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = (1 / (1 + np.exp(-Z)))
        return self.__A

    def cost(self, Y, A):
        """The cost function we will be using"""
        first_term = np.matmul(Y, np.log(A).T)
        second_term = np.matmul((1 - Y), np.log(1.0000001 - A).T)
        loss = first_term + second_term
        return - 1 / A.shape[1] * sum(loss)[0]

    def evaluate(self, X, Y):
        """Evaluates the neurons perdictions"""
        return (np.where(self.forward_prop(X) >= 0.5, 1, 0),
                self.cost(Y, self.forward_prop(X)))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = A.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(X, dZ.T)
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - (alpha * dW.T)
        self.__b = self.__b - alpha * db