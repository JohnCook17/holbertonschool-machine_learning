#!/usr/bin/env python3
"""Learning rate decay in tf"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """alpha the original learning rate. decay_rate the rate alpha decays at.
    global_step, how many times gradient descent has been run. decay_step
    the number of steps before alpha should decay again."""
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
