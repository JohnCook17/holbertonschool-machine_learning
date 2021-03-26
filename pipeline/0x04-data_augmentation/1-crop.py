#!/usr/bin/env python3
"""crops an image using tensorflow"""
import tensorflow as tf


def crop_image(image, size):
    """crops an image, image is the image to crop and size is the new size"""
    return tf.image.random_crop(image, size)
