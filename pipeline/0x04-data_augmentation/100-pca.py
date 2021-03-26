#!/usr/bin/env python3
"""
Bassed off of
https://github.com/koshian2/PCAColorAugmentation
I modified pca_aug_numpy_single.py to fit the parameters
of the task given to me.
"""
import numpy as np


def pca_color(image, alphas):
    """Preforms pca color augmentation"""
    image = image.numpy()

    img = image.reshape(-1, 3).astype(np.float32)
    scaling_factor = np.sqrt(3.0 / np.sum(np.var(img, axis=0)))
    img *= scaling_factor

    cov = np.cov(img, rowvar=False)
    U, S, V = np.linalg.svd(cov)

    delta = np.dot(U, alphas * S)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image + delta, 0, 255).astype(np.uint8)

    return img_out
