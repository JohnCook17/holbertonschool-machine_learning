#!/usr/bin/env python3
"""Saves and loads the weights of a model"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves the weights"""
    network.save_weights(filename, save_format)
    return None


def load_weights(network, filename):
    """loads the weights"""
    network.load_weights(filename)
    return None
