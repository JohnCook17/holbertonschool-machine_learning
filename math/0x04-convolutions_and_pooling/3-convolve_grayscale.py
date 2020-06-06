#!/usr/bin/env python3
"""
A same con, using numpy
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a array with the shape of, the number of images,
    the height of the images, and the width of the images. kernel is
    the filter to apply to these images and is the shape of, kernel height
    kernel width.
    """
    i = images.shape[0]
    i_start = 0
    i_end = kernel.shape[0]
    j_start = 0
    j_end = kernel.shape[1]
    if padding == "same":
        new_array_h = int(images.shape[1] / stride[0])
        new_array_w = int(images.shape[2] / stride[1])
        new_array = np.empty((i, new_array_h, new_array_w))
        p1 = kernel.shape[0] // 2
        if p1 % 2 == 0:
            p1 = kernel.shape[0] // 2
        p2 = kernel.shape[1] // 2
        if p2 % 2 == 0:
            p2 = kernel.shape[1] // 2
        images = np.pad(array=images, pad_width=((0, 0), (p1, p1), (p2, p2)),
                        mode="constant", constant_values=0)
    elif padding == "valid":
        new_array_h = int((images.shape[1] - kernel.shape[0] + 1) / stride[0])
        new_array_w = int((images.shape[2] - kernel.shape[1] + 1) / stride[1])
        new_array = np.empty((i, new_array_h, new_array_w))
    else:
        p1 = padding[0]
        p2 = padding[1]
        new_array_h = int((images.shape[1] + (2 * p1) - kernel.shape[0] + 1) /
                          stride[0])
        new_array_w = int((images.shape[2] + (2 * p2) - kernel.shape[1] + 1) /
                          stride[1])
        new_array = np.empty((i, new_array_h, new_array_w))
        images = np.pad(array=images, pad_width=((0, 0), (p1, p1), (p2, p2)),
                        mode="constant", constant_values=0)
    for i_index in range(0, new_array.shape[1]):
        for j_index in range(0, new_array.shape[2]):
            n = images[:, i_start: i_end, j_start: j_end]
            new_pixel = np.sum((n * kernel), axis=(1, 2))
            new_array[:, i_index, j_index] = new_pixel
            j_start += stride[0]
            j_end += stride[0]
        j_start = 0
        j_end = kernel.shape[1]
        i_start += stride[1]
        i_end += stride[1]
    return new_array
