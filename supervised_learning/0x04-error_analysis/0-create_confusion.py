#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """multiplies the labels tansposed by the logits to create a
    confusion matirx"""
    return np.matmul(labels.T,  logits)
