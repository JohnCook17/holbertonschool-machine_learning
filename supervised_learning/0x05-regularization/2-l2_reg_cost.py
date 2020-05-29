#!/usr/bin/env python3
"""l2 cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """l2 cost in tf"""
    return tf.nn.l2_loss(cost)
