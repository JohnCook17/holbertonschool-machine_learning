#!/usr/bin/env python3
"""Flips an image"""
import tensorflow as tf


def flip_image(image):
    """flips an image, image is an image to flip"""
    return tf.image.flip_left_right(image)
