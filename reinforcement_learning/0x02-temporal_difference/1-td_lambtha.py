#!/usr/bin/env python3
"""temporal dif of lambda"""

import numpy as np
import gym


def td_lambtha(env,
               V,
               policy,
               lambtha,
               episodes=5000,
               max_steps=100,
               alpha=0.1,
               gamma=0.99):
    """Finds the temporal difference of the V values"""
    episode = [[], []]
    Et = [0 for i in range(env.observation_space.n)]
    for i in range(episodes):
        state = env.reset()
        for j in range(max_steps):
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1

            action = policy(state)
            new_state, reward, done, info = env.step(action)

            delta_t = reward + gamma * V[new_state] - V[state]

            V[state] = V[state] + alpha * delta_t * Et[state]

            if done:
                break
            state = new_state
    return np.array(V)
