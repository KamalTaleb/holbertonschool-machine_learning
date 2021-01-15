#!/usr/bin/env python3
"""Contains the NeuralNetwork"""

import numpy as np


class NeuralNetwork:
    """NeuralNetwork class"""

    def __init__(self, nx, nodes):
        """
        constructor
        :param nx: number of input features
        :param nodes: number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        """The weights vector for the hidden layer"""
        self.__W1 = np.random.normal(0, 1, (nodes, nx))

        """The bias for the hidden layer"""
        self.__b1 = np.zeros((nodes, 1))

        """The activated output for the hidden layer"""
        self.__A1 = 0

        """The weights vector for the output neuron"""
        self.__W2 = np.random.normal(0, 1, (1, nodes))

        """The bias for the output neuron"""
        self.__b2 = 0

        """The activated output for the output neuron"""
        self.__A2 = 0

    @property
    def W1(self):
        """weigths to retrieve W1"""
        return self.__W1

    @property
    def b1(self):
        """bias to retrieve b1"""
        return self.__b1

    @property
    def A1(self):
        """activation to retrieve A1"""
        return self.__A1

    @property
    def W2(self):
        """weights to retrieve W2"""
        return self.__W2

    @property
    def b2(self):
        """bias to retrieve b2"""
        return self.__b2

    @property
    def A2(self):
        """activation to retrieve A2"""
        return self.__A2
    
    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X:  is a numpy.ndarray with shape (nx, m)
            that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        :return: private attributes __A1 and __A2, respectively
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2
