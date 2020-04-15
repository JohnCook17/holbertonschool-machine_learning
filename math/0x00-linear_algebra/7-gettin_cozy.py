#!/usr/bin/env python3
"""Concatenates two 2d matrices along a specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """A function that concatenates two 2d matrices along a specific axis"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        new_mat = list(map(lambda v: v[:], mat1[:])) + list(map(lambda v: v[:],
                                                                mat2[:]))
        return new_mat
    elif axis == 1 and len(mat1) == len(mat2):
        i = 0
        new_mat = []
        i = 0
        while i < len(mat2):
            new_mat.append([])
            new_mat[i] += list(map(lambda v1: v1[:] + list(map(lambda v2: v2,
                                                               mat2[i])),
                                   mat1[:]))[i]
            i += 1
        return new_mat
    else:
        return None
