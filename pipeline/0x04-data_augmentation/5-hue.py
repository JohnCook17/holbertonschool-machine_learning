#!/usr/bin/env python3
"""adjust the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """adjust the hue of an image based on delta"""
    return tf.image.adjust_hue(image, delta)
