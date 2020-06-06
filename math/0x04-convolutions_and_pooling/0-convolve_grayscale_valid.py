#!/usr/bin/env python3
import numpy as np
"""A valid convolution"""


def convolve_grayscale_valid(images, kernel):
    """images is a np.ndarray with the shape of, the number of images,
    the height of the images, and the width of the images. kernel is
    the filter to apply to these images and is the shape of, kernel height
    kernel width."""
    i = images.shape[0]
    i_start = 0
    i_end = kernel.shape[0]
    j_start = 0
    j_end = kernel.shape[1]
    new_array_h = (images.shape[1] - kernel.shape[0] + 1)
    new_array_w = (images.shape[2] - kernel.shape[1] + 1)
    new_array = np.empty((i, new_array_h, new_array_w))
    for i_index in range(0, new_array.shape[1]):
        for j_index in range(0, new_array.shape[2]):
            n = images[:, i_start: i_end, j_start: j_end]
            new_pixel = np.sum((n * kernel), axis=(1, 2))
            new_array[:, i_index, j_index] = new_pixel
            j_start += 1
            j_end += 1
        j_start = 0
        j_end = kernel.shape[1]
        i_start += 1
        i_end += 1
    return new_array
