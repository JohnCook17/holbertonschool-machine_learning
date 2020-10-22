#!/usr/bin/env python3
"""Finds the determinant of a square matrix"""


def determinant(matrix):
    """Finds the determinant of a matrix"""
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        if len(matrix[0]) == 1:
            return matrix[0][0]
        elif matrix == [[]]:
            return 1
    if not matrix[0]:
        raise ValueError("matrix must be a square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    # checks for if list of list checks if matrix is 0x0, or if the
    # determinant is the single value in the matrix

    t = 0
    for index in range(len(matrix)):
        arr = [[v for row, v in enumerate(line) if row != index]
               for line in matrix[1:]]
        t += (-1) ** index * matrix[0][index] * determinant(arr)
    return t
