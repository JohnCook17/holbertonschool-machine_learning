#!/usr/bin/env python3
"""Decodes 0 - 9 from one hot"""
import numpy as np


def one_hot_decode(one_hot):
    """Uses argmax to decode one hot"""
    if one_hot.size == 0:
        return None
    try:
        return (np.array([np.argmax(one_hot.T[i])
                          for i in range(0, len(one_hot))
                          if isinstance(i, int)]))
    except (ValueError(), IndexError()):
        return None
