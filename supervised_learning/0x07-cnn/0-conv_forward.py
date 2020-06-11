#!/usr/bin/env python3
"""Forward prop of a convnet"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """A_prev is the prev layer of the net, W is the kernels b is the bias
    activation is the activation function, padding is which padding to use
    and stride is how far to stride in height and width. Height comes first
    """
    img_n = A_prev.shape[0]
    img_h = A_prev.shape[1]
    img_w = A_prev.shape[2]
    img_c_out = W.shape[3]
    ker_h = W.shape[0]
    ker_w = W.shape[1]
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == "same":
        ph = int(np.ceil((img_h - 1) * stride[0] + ker_h - img_h) / 2)
        pw = int(np.ceil((img_w - 1) * stride[1] + ker_w - img_w) / 2)
    elif padding == "valid":
        ph, pw = 0, 0
    img_h_out = int((img_h + 2 * ph - ker_h) / stride[0]) + 1
    img_w_out = int((img_w + 2 * pw - ker_w) / stride[1]) + 1
    out_shape = img_n, img_h_out, img_w_out, img_c_out
    r = np.zeros(shape=out_shape)
    np.pad(r, pad_width=(ph, pw), mode="constant", constant_values=0)

    for h in range(0, img_h_out):
        for w in range(0, img_w_out):
            for channel in range(0, img_c_out):
                n = A_prev[:, h * stride[0]: h * stride[0] + ker_h,
                           w * stride[1]: w * stride[1] + ker_w, :]
                r[:, h, w, channel] = (np.sum((n * W[:, :, :, channel]),
                                              axis=(1, 2, 3)))
    return activation(r + b)
