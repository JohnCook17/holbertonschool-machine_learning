#!/usr/bin/env python3
"""
A same con, using numpy
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images is a array with the shape of, the number of images,
    the height of the images, and the width of the images. kernel is
    the filter to apply to these images and is the shape of, kernel height
    kernel width.
    """
    p1 = (kernel.shape[0] - 1) // 2
    p2 = (kernel.shape[1] - 1) // 2
    i = images.shape[0]
    i_start = 0
    i_end = kernel.shape[0]
    j_start = 0
    j_end = kernel.shape[1]
    new_array_h = (images.shape[1] + 2 * p1 - kernel.shape[0] + 1)
    new_array_w = (images.shape[2] + 2 * p2 - kernel.shape[1] + 1)
    new_array = np.empty((i, new_array_h, new_array_w))
    images = np.pad(array=images, pad_width=((0, 0), (p1, p1), (p2, p2)),
                    mode="constant", constant_values=0)
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
