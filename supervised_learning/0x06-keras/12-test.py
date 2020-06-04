#!/usr/bin/env python3
"""Test a model in keras"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a model in keras, network is the network to test
    data is the input data, labels are the correct labels of the
    data, verbose is whether to print or not"""
    evaluation = network.evaluate(x=data, y=labels, verbose=verbose)
    return evaluation
