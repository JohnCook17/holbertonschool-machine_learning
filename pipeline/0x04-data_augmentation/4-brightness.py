#!/usr/bin/env python3
"""Adjust the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """adjust the brightness of an image based on max_delta"""
    return tf.image.adjust_brightness(image, max_delta)
