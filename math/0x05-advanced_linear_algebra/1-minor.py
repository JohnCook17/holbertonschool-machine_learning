#!/usr/bin/env python3
"""Finds the Minors of a square matrix"""
import numpy as np


def minor(matrix):
    """finds all the minors of a matrix"""
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if matrix and isinstance(matrix[0], list):
        for element in matrix:
            if not isinstance(element, list):
                raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a square matrix")
    mat = np.asarray(matrix)
    return [[int(round(np.linalg.det(np.delete(np.delete(mat, i, axis=0),
                                               j,
                                               axis=1))))
             for j in range(len(matrix[i]))]
            for i in range(len(matrix))]
