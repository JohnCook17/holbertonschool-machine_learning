#!/usr/bin/env python3
"""Transposes a matrix"""


def matrix_transpose(matrix):
    """Function that transposes a matrix"""
    new_matrix = [[matrix[j][i] for j in range(len(matrix))]
                  for i in range(len(matrix[0]))]
    return new_matrix
