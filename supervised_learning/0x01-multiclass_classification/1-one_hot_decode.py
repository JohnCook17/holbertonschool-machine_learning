#!/usr/bin/env python3
"""Decodes 0 - 9 from one hot"""
import numpy as np


def one_hot_decode(one_hot):
    """Uses argmax to decode one hot"""
    if not isinstance(one_hot, np.ndarray):
        return None
    if not one_hot.ndim == 2:
        return None
    decode = []
    for i in range(0, len(one_hot)):
        decode.append(np.argmax(one_hot.T[i]))
    return np.array(decode)
