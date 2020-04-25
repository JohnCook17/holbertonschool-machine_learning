#!/usr/bin/env python3
"""Sums and squares n"""


def summation_i_squared(n):
    """uses math to sum the squares"""
    if isinstance(n, float) and n % 1 == 0:
        n = int(n)
        answer = ((n ** 3) / 3) + ((n ** 2) / 2) + (n / 6)
        if answer % 1 == 0:
            answer = int(answer)
        return answer
    elif isinstance(n, int):
        answer = ((n ** 3) / 3) + ((n ** 2) / 2) + (n / 6)
        if answer % 1 == 0:
            answer = int(answer)
        return answer
    else:
        return None
