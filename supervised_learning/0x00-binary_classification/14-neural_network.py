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
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for weights 1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for bias 1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for activated outputs 1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for weights 2"""
        return self.__W2

    @property
    def b2(self):
        """getter for bias 2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for activated outputs 2"""
        return self.__A2

    def forward_prop(self, X):
        """Forward prop for the neural network"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = (1 / (1 + np.exp(-Z1)))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = (1 / (1 + np.exp(-Z2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Cost of the neural network"""
        first_term = np.matmul(Y, np.log(A).T)
        second_term = np.matmul((1 - Y), np.log(1.0000001 - A).T)
        loss = first_term + second_term
        return - 1 / A.shape[1] * sum(loss)[0]

    def evaluate(self, X, Y):
        """The evaluation of the neural network"""
        return (np.where(self.forward_prop(X)[1] >= 0.5, 1, 0),
                self.cost(Y, self.forward_prop(X)[1]))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Gradient descent of the neural network"""
        m = A2.shape[1]
        dZ2 = A2 - Y
        dW2 = 1 / m * np.matmul(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = 1 / m * np.matmul(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        iteration = 0
        while iteration < iterations:
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            iteration += 1
        return self.evaluate(X, Y)
