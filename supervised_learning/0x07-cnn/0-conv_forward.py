#!/usr/bin/env python3
""""""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """"""
    img_n = A_prev.shape[0]
    img_h = A_prev.shape[1]
    img_w = A_prev.shape[2]
    img_c_in = W.shape[2]
    img_c_out = W.shape[3]
    ker_h = W.shape[0]
    ker_w = W.shape[1]
    if isinstance(padding, tuple):
        p1, p2 = padding
    elif padding == "same":
        p1 = int(np.ceil((img_h - 1) * stride[0] + ker_h - img_h) / 2)
        p2 = int(np.ceil((img_w - 1) * stride[1] + ker_w - img_w) / 2)
    elif padding == "valid":
        p1, p2 = 0, 0
    img_h_out = int((img_h + 2 * p1 - ker_h) / stride[0]) + 1
    img_w_out = int((img_w + 2 * p2 - ker_w) / stride[1]) + 1
    out_shape = img_n, img_h_out, img_w_out, img_c_out
    r = np.zeros(shape=out_shape)
    np.pad(r, pad_width=(p1, p2), mode="constant", constant_values=0)

    for h in range(0, img_h_out):
        for w in range(0, img_w_out):
            for channel in range(0, img_c_out):
                n = A_prev[:, h: h + ker_h, w: w + ker_w, :]
                r[:, h, w, channel] = (np.sum((n * W[:, :, :, channel]),
                                              axis=(1, 2, 3)))
    return activation(r + b)
