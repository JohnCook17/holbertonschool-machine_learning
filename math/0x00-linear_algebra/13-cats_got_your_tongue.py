#!/usr/bin/env python3
"""Concatenates two arrays"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """uses np.concatenate to concatenate two arrays along a specified axis"""
    return np.concatenate((mat1, mat2), axis=axis)
