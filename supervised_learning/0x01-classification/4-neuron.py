#!/usr/bin/env python3
"""Contains the Neuron class"""

import numpy as np


# Miscellaneous functions
def sigmoid(Z):
    """sigmoid function"""
    return 1.0 / (1.0 + np.exp(-Z))


class Neuron:
    """the number of input features to the neuron"""

    def __init__(self, nx):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """weights to retrieve it"""
        return self.__W

    @property
    def b(self):
        """bias to retrieve it"""
        return self.__b

    @property
    def A(self):
        """activation to retrieve it"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X"""
        Z = np.matmul(self.__W, X) + self.__b
        """self.__A = 1 / (1 + np.exp(-Z))"""
        self.__A = sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions X"""
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost
