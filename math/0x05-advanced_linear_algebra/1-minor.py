#!/usr/bin/env python3
"""Finds the Minors of a square matrix"""


def d(A):
    # print("A = ", A, type(A))

    if len(A) == 1:
        return A[0][0]
    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    else:
        det = 0
        for i in range(len(A)):
            my_copy = [[value for value in row] for row in A]
            mul = my_copy[0][i]
            my_copy.pop(0)
            for ele in my_copy:
                ele.pop(i)
            ret = d(my_copy)
            det += mul * ret * (-1) ** i
        return det


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
    if len(matrix) == 1:
        return [[1]]
    minors = []
    for i in range(len(matrix)):
        minors.append([])
        for j in range(len(matrix)):
            minors[i].append(d([row[:j] + row[j + 1:]
                                for row in (matrix[:i] + matrix[i + 1:])]))

    return minors
