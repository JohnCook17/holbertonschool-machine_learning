#!/usr/bin/env python3
"""Performst math operations on arrays"""


def np_elementwise(mat1, mat2):
    """adds, subtracts, multiplies and divides two arrays element wise"""
    mat_add = mat1 + mat2
    mat_sub = mat1 - mat2
    mat_mul = mat1 * mat2
    mat_div = mat1 / mat2
    return (mat_add, mat_sub, mat_mul, mat_div)
