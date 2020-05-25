#!/usr/bin/env python3
"""Calculates precision of confusion matrix"""
import numpy as np


def precision(confusion):
    """calculates precision of confusion matrix by taking the true
    positives and false positives and finding the precision"""
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    pre = true_positives / (true_positives + false_positives)
    return pre
