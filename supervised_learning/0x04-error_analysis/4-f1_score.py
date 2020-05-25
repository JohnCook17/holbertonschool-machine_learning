#!/usr/bin/env python3
"""F1 score of a confusion matrix"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Finds the f1 score, or harmonic mean of precision and sensitivity"""
    pre = precision(confusion)
    sen = sensitivity(confusion)
    return np.asarray(2 * (pre * sen) / (pre + sen))
