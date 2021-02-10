#!/usr/bin/env python3
""""""
import numpy as np


def q_init(env):
    """"""
    return np.zeros([env.observation_space.n, env.action_space.n])
