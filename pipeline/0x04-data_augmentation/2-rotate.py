#!/usr/bin/env python3
"""Rotates an image"""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image 90 degrees counter clock wise"""
    return tf.image.rot90(image)
