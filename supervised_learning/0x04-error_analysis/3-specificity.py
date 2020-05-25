#!/usr/bin/env python3
""""""
import numpy as np


def specificity(confusion):
    """"""
    classes = confusion.shape[0]
    true_negative = []
    for i in range(classes):
        temp = np.delete(confusion, i, 0)
        temp = np.delete(temp, i, 1)
        true_negative.append(sum(sum(temp)))
    true_negative = np.asarray(true_negative)
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    spe = true_negative / (true_negative + false_positives)
    return spe
