#!/usr/bin/env python3
"""Takes a polynomial in the the form of a list and integrates it"""


def poly_integral(poly, C=0):
    """returns the result of integration of a polynomial"""
    try:
        res = [C]
        for i in range(len(poly)):
            c = poly[i] / (i + 1)
            if c % 1 == 0:
                c = int(c)
            res.append(c)
        return res
    except(ValueError, IndexError):
        return None
