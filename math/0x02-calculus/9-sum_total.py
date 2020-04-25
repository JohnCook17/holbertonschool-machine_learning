#!/usr/bin/env python3
"""Sums and squares n"""


def summation_i_squared(n):
    """uses math to sum the squares"""
    if type(n) is int or type(n) is float:
        return ((n ** 3) / 3) + ((n ** 2) / 2) + (n / 6)
    return None
