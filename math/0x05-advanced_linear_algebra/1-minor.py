#!/usr/bin/env python3
"""finds the minor of a matrix"""


def minor(matrix):
    """finds the minor of a matrix"""
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")
    for element in matrix:
        if not isinstance(element, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        if len(matrix[0]) == 1:
            return [[1]]
        elif matrix == [[]]:
            raise ValueError("matrix must be a non-empty square matrix")
    if not matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    def determinant(matrix):
        """Finds the determinant of a matrix"""
        t = 0
        for index in range(len(matrix)):
            arr = [[v for row, v in enumerate(line) if row != index]
                   for line in matrix[1:]]
            t += (-1) ** index * matrix[0][index] * determinant(arr)
        return t

    def get_sub_matrix(matrix, i, j):
        """gets the sub matrix"""
        length = len(matrix)
        # print(length)
        m = [[col for col in row] for row in matrix]
        for row in range(length):
            # print("row = ", row)
            if row == i:
                del m[row]
            for col in range(length):
                # print("col = ", col)
                if row == length - 1:
                    return m
                # print("===", m, "===")
                if col == j:
                    del m[row][col]
                # print("===", m, "===")
        return m

    length = len(matrix)
    minors = []
    for i in range(length):
        row = []
        for j in range(length):
            # print(get_sub_matrix(matrix, i, j))
            col = determinant(get_sub_matrix(matrix, i, j))
            # print(col)
            row.append(col)
        minors.append(row)

    return minors
