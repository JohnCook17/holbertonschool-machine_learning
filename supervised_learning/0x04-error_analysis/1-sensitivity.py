#!/usr/bin/env python3
"""Calculates the sensitivity or recall of a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """"""
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    sen = true_positives / (true_positives + false_negatives)
    return sen
