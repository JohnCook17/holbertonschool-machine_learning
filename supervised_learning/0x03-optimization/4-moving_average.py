#!/usr/bin/env python3
"""Exponential weighted moving average with bias correction"""
import numpy as np


def moving_average(data, beta):
    """Corrects for bias in an Exponential weighted moving average"""
    vt = 0
    ewma = []
    for i in range(0, (len(data))):
        bt = (1 - beta ** (i + 1))
        vt = ((beta * vt) + ((1 - beta) * data[i]))
        ewma.append(vt / bt)
    return ewma
