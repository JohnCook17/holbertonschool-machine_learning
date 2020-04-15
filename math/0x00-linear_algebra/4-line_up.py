#!/usr/bin/env python3
"""Adds two arrays together"""


def add_arrays(arr1, arr2):
    """A function that adds two arrays together"""
    if len(arr1) == len(arr2):
        return [i + j for i, j in zip(arr1, arr2)]
    else:
        return None
