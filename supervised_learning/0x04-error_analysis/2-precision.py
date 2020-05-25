#!/usr/bin/env python3
""""""
import numpy as np


def precision(confusion):
    """"""
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    pre = true_positives / (true_positives + false_positives)
    return pre
