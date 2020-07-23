#!/usr/bin/env python3
"""Finds the Determinant of a square matrix"""


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
            # my_copy = [ele.pop(i) for ele in my_copy]
            for ele in my_copy:
                ele.pop(i)
            ret = d(my_copy)
            det += mul * ret * (-1) ** i
        return det
    """return (sum((-1) ** i * A[0][i] * d([[A[x][y]for x in range(1, len(A))]
                                        for y in range(1 + i, len(A))])
                for i in range(len(A)))if isinstance(A, list) and len(A) > 1
            else A[0][0])"""


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
    return d(matrix)
