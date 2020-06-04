#!/usr/bin/env python3
"""Saves and loads a model from json"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves the network config"""
    with open(filename, "wt") as my_file:
        my_file.write(network.to_json())
    return None


def load_config(filename):
    """loads the network config"""
    with open(filename, "r") as my_file:
        my_file = my_file.read()
        return K.models.model_from_json(my_file)
