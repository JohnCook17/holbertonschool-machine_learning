#!/usr/bin/env python3
"""Determines the definiteness of a matirx"""
import numpy as np


def definiteness(matrix):
    """calculates Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    try:
        if np.where(np.linalg.eigvalsh(matrix) > 0,
                    True, False).all() is np.bool_(True):
            return "Positive definite"
        elif np.where(np.linalg.eigvalsh(matrix) >= 0,
                      True, False).all() is np.bool_(True):
            return "Positive semi-definite"
        elif np.where(np.linalg.eigvalsh(matrix) < 0,
                      True, False).all() is np.bool_(True):
            return "Negative definite"
        elif np.where(np.linalg.eigvalsh(matrix) <= 0,
                      True, False).all() is np.bool_(True):
            return "Negative semi-definite"
        else:
            return "Indefinite"
    except Exception as e:
        return None
