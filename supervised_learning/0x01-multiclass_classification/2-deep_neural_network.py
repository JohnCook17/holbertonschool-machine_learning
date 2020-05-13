#!/usr/bin/env python3
"""A deep neural network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def forward_prop(self, X):
        """Forward prop for the deep neural network using sigmoid"""
        nx = X.shape[0]
        m = X.shape[1]
        self.__cache["A0"] = X
        for value in range(1, self.__L + 1):
            Z = (np.matmul(self.__weights["W" + str(value)],
                           self.__cache["A" + str(value - 1)]) +
                 self.__weights["b" + str(value)])
            self.__cache["A" + str(value)] = (1 / (1 + np.exp(-Z)))
        return self.__cache["A" + str(value)], self.__cache

    def cost(self, Y, A):
        """The cost of the deep neural network"""
        first_term = np.matmul(Y, np.log(A).T)
        second_term = np.matmul((1 - Y), np.log(1.0000001 - A).T)
        loss = first_term + second_term
        return - 1 / A.shape[1] * sum(loss)[0]

    def evaluate(self, X, Y):
        """The evaluation of the deep neural network"""
        return (np.where(self.forward_prop(X)[0] >= 0.5, 1, 0),
                self.cost(Y, self.forward_prop(X)[0]))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """The gradient_descent for the deep neural network"""
        m = Y.shape[1]
        for value in range(self.__L, 0, -1):
            if value == self.__L:
                dZ = cache["A" + str(value)] - Y
            else:
                dZ = dA * (cache["A" + str(value)] * (1 - cache["A" +
                                                                str(value)]))
            dW = 1 / m * np.matmul(dZ, cache["A" + str(value - 1)].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(self.__weights["W" + str(value)].T, dZ)
            self.__weights["W" + str(value)] = (self.__weights["W" +
                                                               str(value)] -
                                                alpha * dW)
            self.__weights["b" + str(value)] = (self.__weights["b" +
                                                               str(value)] -
                                                alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network with forward prop then
        gradient descent"""
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
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and iteration % step == 0:
                print("Cost after {} iterations: {}"
                      .format(iteration, self.cost(Y, A)))
            if graph and iteration % step == 0:
                x_info.append(iteration)
                info.append(self.cost(Y, A))
            iteration += 1
        if verbose:
            print("Cost after {} iterations: {}"
                  .format(iteration,
                          self.cost(Y, self.forward_prop(X)[0])))
        if graph:
            x_info.append(iteration)
            info.append(self.cost(Y, A))
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.plot(x_info, info, "b-")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves a pickle of the deep neural network"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, "wb") as filename:
            pickle.dump(self, filename)

    @staticmethod
    def load(filename):
        """Loads a pickle of the deep neural network"""
        if not filename:
            return None
        if not isinstance(filename, str):
            return None
        if filename[-4:] != ".pkl":
            return None
        with open(filename, "rb") as myfile:
            return pickle.load(myfile)
