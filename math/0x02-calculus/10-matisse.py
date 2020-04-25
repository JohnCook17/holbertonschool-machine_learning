#!/usr/bin/env python3
"""Finds the derivative of an equation"""


def poly_derivative(poly):
    """Finds a derivative of an equation"""
    try:
        answer = [poly[i] * i for i in range(1, len(poly))]
        if answer == 0:
            return [0]
        else:
            return answer
    except (ValueError, IndexError):
        return None
