#!/usr/bin/env python3
"""Finds the Determinant of a square matrix"""
import numpy as np


def determinant(matrix):
    """Finds the determinate of a square matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if matrix and isinstance(matrix[0], list):
        for element in matrix:
            if not isinstance(element, list):
                raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    matrix = np.asarray(matrix)
    return int(round(np.linalg.det(matrix)))
