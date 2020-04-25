#!/usr/bin/env python3
"""Sums and squares n"""


def summation_i_squared(n):
    """uses math to sum the squares"""
    if type(n) == int or type(n) == float:
        answer = ((n ** 3) / 3) + ((n ** 2) / 2) + (n / 6)
        if answer % 1 == 0:
            answer = int(answer)
        return answer
    return None