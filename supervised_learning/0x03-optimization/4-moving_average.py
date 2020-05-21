#!/usr/bin/env python3
""""""
import numpy as np


def moving_average(data, beta):
    """"""
    ewma = [data[0]]
    for i in range(1, (len(data))):
        bt = (1 - beta ** i)
        vt = data[i] / bt
        print(vt)
        ewma.append((beta * ewma[i - 1]) + ((1 - beta) * vt))
    return ewma
