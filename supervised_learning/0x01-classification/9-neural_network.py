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
        self.W1 = np.random.normal(0, 1, (nodes, nx))

        """The bias for the hidden layer"""
        self.b1 = np.zeros((nodes, 1))

        """The activated output for the hidden layer"""
        self.A1 = 0

        """The weights vector for the output neuron"""
        self.W2 = np.random.normal(0, 1, (1, nodes))

        """The bias for the output neuron"""
        self.b2 = 0

        """The activated output for the output neuron (prediction)"""
        self.A2 = 0

    @property
    def W1(self):
        """weights to retrieve W1"""
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
