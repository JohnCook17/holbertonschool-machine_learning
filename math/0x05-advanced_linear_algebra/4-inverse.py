#!/usr/bin/env python3
"""Finds the inverse of a square matrix"""
import numpy as np
# each element of the adjugate divided by the determinat

def inverse(matrix):
    """finds the inverse of a matrix"""
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
    matrix = np.asarray(matrix)
    try:
        return np.round(np.linalg.inv(matrix), decimals=15).tolist()
    except Exception as e:
        return None
