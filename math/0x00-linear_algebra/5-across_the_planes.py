#!/usr/bin/env python3
"""Adds two matrices together"""


def add_matrices2D(mat1, mat2):
    """A function that adds two matrices together"""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        return [[colum1 + colum2 for colum1, colum2 in zip(row1, row2)]
                for row1, row2 in zip(mat1, mat2)]
    else:
        return None
