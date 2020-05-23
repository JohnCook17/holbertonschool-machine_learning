#!/usr/bin/env python3
"""Dynamic adjustment of learning rate."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """alpha the original learning rate. decay_rate determines how fast
    alpha decays. global_step is how many times gradient descent has been
    performed. decay_step is the number of passes before alpha should decay"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
