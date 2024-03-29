#!/usr/bin/env python3
"""Finds the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Finds the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    for element in matrix:
        if not isinstance(element, np.ndarray):
            raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        if np.all(np.linalg.eigvals(matrix) == 0):
            return "Indefinite"
        if np.all(np.linalg.eigvals(matrix) > 0):
            return "Positive definite"
        elif np.all(np.linalg.eigvals(matrix) < 0):
            return "Negative definite"
        elif np.all(np.linalg.eigvals(matrix) >= 0):
            return "Positive semi-definite"
        elif np.all(np.linalg.eigvals(matrix) <= 0):
            return "Negative semi-definite"
        else:
            return "Indefinite"
    except ZeroDivisionError:
        return None
