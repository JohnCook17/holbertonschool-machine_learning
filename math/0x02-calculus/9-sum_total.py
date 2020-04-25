#!/usr/bin/env python3
"""Sums and squares n"""


def summation_i_squared(n):
    """uses math to sum the squares"""
    try:
        if isinstance(n, int):
            if n <= 0:
                return None
            if n == 1:
                return 1
            if n > 0:
                answer = ((n ** 3) / 3) + ((n ** 2) / 2) + (n / 6)
                return int(answer)
        else:
            return None
    except ValueError:
        return None
