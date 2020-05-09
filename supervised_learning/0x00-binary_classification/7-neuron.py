#!/usr/bin/env python3
"""A Neuron for binary classification"""
import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        iteration = 0
        if graph:
            x_info = []
            info = []
        while iteration < iterations:
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if verbose and iteration % step == 0:
                print("Cost after {} iterations: {}"
                      .format(iteration, self.cost(Y, A)))
            if graph and iteration % step == 0:
                x_info.append(iteration)
                info.append(self.cost(Y, A))
            iteration += 1
        if verbose:
            print("Cost after {} iterations: {}"
                  .format(iteration, self.cost(Y, self.forward_prop(X))))
        if graph:
            x_info.append(iteration)
            info.append(self.cost(Y, A))
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.plot(x_info, info, "b-")
            plt.show()
        return self.evaluate(X, Y)
