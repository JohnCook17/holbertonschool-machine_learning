#!/usr/bin/env python3
"""Batch normalization"""


def batch_norm(Z, gamma, beta, epsilon):
    """Z is the np.ndarray to normalize. gamma is a np.ndarray
    used to scale the batch. beta is a np.ndarray containing the
    offsets for the batch. epsilon avoids zero division"""
    m, n = Z.shape
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    std = (variance + epsilon) ** .5
    cent = Z - mean
    norm = cent / std
    return gamma * norm + beta
