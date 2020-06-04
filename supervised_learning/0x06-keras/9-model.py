#!/usr/bin/env python3
"""saves and loads a model"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves the model"""
    network.save(filename)
    return None


def load_model(filename):
    """loads model"""
    model = K.models.load_model(filename)
    return model
