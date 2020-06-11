#!/usr/bin/env python3
"""Pooling layer forward prop"""
import numpy as np


def asStride(arr, sub_shape, stride):
    """Get a strided sub-matrices view of an ndarray.
    using stride_tricks"""
    img_s = arr.strides[0]
    s0, s1 = arr.strides[1], arr.strides[2]
    img_n = arr.shape[0]
    m1, n1 = arr.shape[1], arr.shape[2]
    m2, n2 = sub_shape
    view_shape = (img_n, 1 + (m1-m2) // stride[0], 1 + (n1 - n2) //
                  stride[1], m2, n2) + arr.shape[3:]
    strides = (img_s, stride[0] * s0, stride[1] * s1, s0, s1, arr.strides[3])
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """A_prev is the previous image in the cnn, kernel_shape is the
    kernel_shape, stride is how far to stride in height and width,
    mode is which mode to use."""
    img_n = A_prev.shape[0]
    img_h = A_prev.shape[1]
    img_w = A_prev.shape[2]
    img_c = A_prev.shape[3]
    ker_h = kernel_shape[0]
    ker_w = kernel_shape[1]

    img_h_out = (img_h - ker_h) // stride[0] * stride[0] + ker_h
    img_w_out = (img_w - ker_w) // stride[1] * stride[1] + ker_w
    pad = A_prev[:, :img_h_out * ker_h, :img_w_out * ker_w, ...]

    output = asStride(pad, kernel_shape, stride)
    if mode == "max":
        r = np.nanmax(output, axis=(3, 4))
    else:
        r = np.nanmean(output, axis=(3, 4))
    return r
