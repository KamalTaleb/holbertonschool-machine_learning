#!/usr/bin/env python3
"""the Neuron class"""

import numpy as np


class Neuron:
    """the number of input features to the neuron"""

    def __init__(self, n):
        """constructor"""
        if not isinstance(n, int):
            raise TypeError("the number should be an integer")
        if n < 1:
            raise ValueError("the number should be a positive integer")
        self.W = np.random.normal(0, 1, (1, n))
        self.b = 0
        self.A = 0
