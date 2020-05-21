#!/usr/bin/env python3
"""Normalizes data"""
import numpy as np


def normalization_constants(X):
    """Takes the normalization and standardization of X"""
    return np.mean(X, axis=0), np.std(X, axis=0)
