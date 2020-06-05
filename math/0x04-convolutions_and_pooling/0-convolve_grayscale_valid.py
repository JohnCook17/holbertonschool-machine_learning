#!/usr/bin/env python3
import numpy as np
"""A valid convolution"""


def convolve_grayscale_valid(images, kernel):
    """images is a np.ndarray with the shape of, the number of images,
    the height of the images, and the width of the images. kernel is
    the filter to apply to these images and is the shape of, kernel height
    kernel width. d1 is how many pixels out from the center of the kernel in
    dimension 1, d2 is how many pixels out from the center of the kernel in
    dimension 2. imgs is the number of images."""
    d1 = kernel.shape[0] // 2
    d2 = kernel.shape[1] // 2
    imgs = images.shape[0]
    new_array_w = (images.shape[1] - (2 * d1))
    new_array_h = (images.shape[2] - (2 * d2))
    new_array = np.empty((imgs, new_array_w, new_array_h))
    for i_index in range(1, new_array.shape[1]):
        for j_index in range(1, new_array.shape[2]):
            i_start = i_index - d1
            i_end = i_index + d1 + 1
            j_start = j_index - d2
            j_end = j_index + d2 + 1
            n = images[:, i_start: i_end, j_start: j_end]
            new_pixel = np.sum((n * kernel), axis=(1, 2))
            new_array[:, i_index, j_index] = new_pixel
    return new_array
