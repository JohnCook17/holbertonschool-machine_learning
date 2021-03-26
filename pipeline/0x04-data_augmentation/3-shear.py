#!/usr/bin/env python3
"""Shears an image"""
import tensorflow as tf


def shear_image(image, intensity):
    """Shears an image based on intensity"""
    image = tf.expand_dims(image, axis=0)
    datagen = (tf.keras.preprocessing.image.
               ImageDataGenerator(shear_range=intensity))
    datagen.fit(x=image)
    new_image = datagen.flow(image)
    new_image = new_image.next()[0].astype("int")
    return new_image
