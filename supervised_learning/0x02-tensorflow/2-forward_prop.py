#!/usr/bin/env python3
"""Forward prop with tf"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """x is the placeholder for input, layer_sizes is a list of layer sizes,
    and activations is a list of activation functions to use"""
    prev = x
    for layer, activ in zip(layer_sizes, activations):
        prev = create_layer(prev, layer, activ)
    return prev
