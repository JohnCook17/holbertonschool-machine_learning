#!/usr/bin/env python3
""""""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """"""
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

    dA_prev = np.zeros(shape=(img_n, img_h_out, img_w_out, img_c_out))
    pad_A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph,), (pw, pw), (0, 0)),
                 mode="constant", constant_values=0)
    pad_dZ = np.pad(dZ, pad_width=((0, 0), (ph, ph,), (pw, pw), (0, 0)),
                 mode="constant", constant_values=0)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    print(img_h, img_w)
    print(img_h_out, img_w_out)
    print(ker_h, ker_w)
    print(pad_dZ.shape)

    for h in range(0, img_h_out):
        for w in range(0, img_w_out):
            for c in range(0, img_c_out):
                print("loop")
                print(h, w, c)
                try:
                    dA_slice = dA_prev[:, h * stride[0]: h * stride[0] + ker_h,
                                    w * stride[1]: w * stride[1] + ker_w, :]
                    dZ_slice = pad_dZ[:, h * stride[0]: h * stride[0] + ker_h,
                                w * stride[1]: w * stride[1] + ker_w, :]
                    dA_slice += W[:, :, :, c] * dZ_slice
                    # dW += [:, h * stride[0]: h * stride[0] + ker_h, w * stride[1]: w * stride[1] + ker_w, :] * dZ_slice
                except ValueError:
                    pass
    db = np.sum(dZ, axis=(1, 2, 3))
    print(dA_prev.shape)
    return dA_prev, dW, db
