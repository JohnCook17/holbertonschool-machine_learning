#!/usr/bin/env python3
"""A simple Markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """P is the probability and s is the initial state"""
    for _ in range(t):
        s = s @ P
    return s
