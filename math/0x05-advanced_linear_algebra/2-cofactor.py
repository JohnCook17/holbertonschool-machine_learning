#!/usr/bin/env python3
"""Finds the cofactor of a square matrix"""
import numpy as np


def cofactor(matrix):
    """finds the cofactor of a matrix"""
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
    U, sigma, Vt = np.linalg.svd(matrix)
    N = len(sigma)
    g = np.tile(sigma, N)
    g[::(N+1)] = 1
    G = np.diag(-(-1)**N*np.product(np.reshape(g, (N, N)), 1))
    ret_mat = np.rint(U @ G @ Vt).tolist()
    ret_mat = [[int(round(j)) for j in i] for i in ret_mat]
    return ret_mat
