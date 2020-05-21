#!/usr/bin/env python3
""""""
import numpy as np


def moving_average(data, beta):
    """"""
    prev = data[0]
    ewma = []
    for i in range((len(data))):
        ewma.append(((1 - beta) * prev) + (beta * data[i]))
        prev = data[i]
    return ewma
