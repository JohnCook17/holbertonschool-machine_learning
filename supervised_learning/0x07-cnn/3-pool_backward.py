#!/usr/bin/env python3
"""placeholder for 3"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """dZ is the derivative of Z, A_prev is the previous layer, W is the kernel
    b is the bias, padding can be a tuple, same or valid, stride is how far to
    stride"""
    img_n = dZ.shape[0]
    img_h = A_prev.shape[1]
    img_w = A_prev.shape[2]
    img_c_in = A_prev.shape[3]
    img_c_out = dZ.shape[3]

    ker_h = W.shape[0]
    ker_w = W.shape[1]

    dZ_h = dZ.shape[1]
    dZ_w = dZ.shape[2]

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == "same":
        ph = int(np.ceil((img_h - 1) * stride[0] + ker_h - img_h) / 2)
        pw = int(np.ceil((img_w - 1) * stride[1] + ker_w - img_w) / 2)
    elif padding == "valid":
        ph, pw = 0, 0
    img_h_out = int((img_h + 2 * ph - ker_h) / stride[0]) + 1
    img_w_out = int((img_w + 2 * pw - ker_w) / stride[1]) + 1

    pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph,), (pw, pw), (0, 0)),
                 mode="constant", constant_values=0)
    dW = np.zeros(W.shape)
    dA = np.zeros(pad.shape)
    # print(img_h, img_w)
    # print(img_h_out, img_w_out)
    # print(ker_h, ker_w)
    # print(dZ.shape)

    for i in range(0, img_n):
        for h in range(0, img_h_out):
            for w in range(0, img_w_out):
                for c in range(0, img_c_out):
                    # print("loop")
                    # print(i, h, w, c)
                    dA_slice = dA[i, h * stride[0]: h * stride[0] + ker_h,
                                  w * stride[1]: w * stride[1] + ker_w, :]
                    Ap_s = pad[i, h * stride[0]: h * stride[0] + ker_h,
                               w * stride[1]: w * stride[1] + ker_w, :]
                    dA_slice += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += Ap_s * dZ[i, h, w, c]
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    return dA, dW, db
