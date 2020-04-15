#!/usr/bin/env python3
"""Matrix multiplication"""


def mat_mul(mat1, mat2):
    """A function that performs matrix multiplication"""
    if len(mat1[0]) == len(mat2):
        return [[sum([x*y for (x, y) in zip(row, col)]) for col in zip(*mat2)]
                for row in mat1]
