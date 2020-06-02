#!/usr/bin/env python3
"""one hot encoding using keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """labels is the labels of the data,
    classes is the number of data classes"""
    return K.utils.to_categorical(labels, num_classes=classes)
