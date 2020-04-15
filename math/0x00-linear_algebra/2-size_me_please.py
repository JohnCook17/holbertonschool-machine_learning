#!/usr/bin/env python3
"""Finds the size of an array"""


def matrix_shape(matrix):
    """A function that finds the size of an array"""
    shape = []

    def recursive_helper(matrix, shape):
        """A helper function because I need shape"""
        if type(matrix) is int:
            return shape
        shape.append(len(matrix))
        return recursive_helper(matrix[0], shape)
    return recursive_helper(matrix, shape)
