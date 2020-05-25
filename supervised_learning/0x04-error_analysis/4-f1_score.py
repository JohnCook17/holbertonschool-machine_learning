#!/usr/bin/env python3
""""""
import numpy as np


def f1_score(confusion):
    """"""
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    recall = true_positives / (true_positives + false_negatives)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precission = true_positives / (true_positives + false_positives)
    return 2 * (precission * recall) / (precission + recall)
