#!/usr/bin/env python3
"""the definiteness method"""

import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix"""
    # type test
    err = 'matrix must be a numpy.ndarray'
    if not isinstance(matrix, np.ndarray):
        raise TypeError(err)

    # square test
    my_len = matrix.shape[0]
    if len(matrix.shape) != 2 or my_len != matrix.shape[1]:
        return None

    # symmetry test
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    # eigenvalues
    w, v = np.linalg.eig(matrix)

    if all(w > 0):
        return 'Positive definite'
    elif all(w >= 0):
        return 'Positive semi-definite'
    elif all(w < 0):
        return 'Negative definite'
    elif all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
