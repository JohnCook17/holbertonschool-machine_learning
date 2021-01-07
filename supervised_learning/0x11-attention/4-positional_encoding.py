#!/usr/bin/env python3
"""Encodes the sentence so that it works with Attention"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Positions the words"""
    def cal_angle(position, hid_idx):
        """Calculates the angle of the position"""
        return position / np.power(10000, 2 * (hid_idx // 2) / dm)

    def get_position_vec(position):
        """makes a position vector"""
        return [cal_angle(position, i) for i in range(dm)]

    f = np.array([get_position_vec(j) for j in range(max_seq_len)])

    f[:, 0::2] = np.sin(f[:, 0::2])
    f[:, 1::2] = np.cos(f[:, 1::2])

    return f
