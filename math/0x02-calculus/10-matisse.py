#!/usr/bin/env python3
"""Finds the derivative of an equation"""


def poly_derivative(poly):
    """Finds a derivative of an equation"""
    if [n for n in poly if type(n) == int or type(n) == float]:
        answer = [poly[i] * i for i in range(1, len(poly))]
        if answer == 0:
            return [0]
        else:
            return answer
    return None
