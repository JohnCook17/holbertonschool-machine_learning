#!/usr/bin/env python3
"""Sums and squares n"""


def summation_i_squared(n):
    """uses math to sum the squares"""
    try:
        if isinstance(n, int):
            if n < 0:
                return None
            if n >= 0:
                answer = ((n ** 3) / 3) + ((n ** 2) / 2) + (n / 6)
                return int(answer)
    except ValueError:
        return None
