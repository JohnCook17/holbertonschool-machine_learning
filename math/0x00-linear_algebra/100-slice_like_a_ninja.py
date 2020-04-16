#!/usr/bin/env python3
"""Uses the built in slice function to slice an np array"""


def np_slice(matrix, axes={}):
    """Slices the np array with the key values stored in axes"""
    for key, value in axes.items():
        args = axes.get(key)
        array_slice = slice(*args)
        if key == 0:
            new_array = matrix[array_slice]
        elif key == 1:
            new_array = matrix.T[array_slice]
    return new_array
