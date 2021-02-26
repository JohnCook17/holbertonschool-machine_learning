#!/usr/bin/env python3
"""The monte carlo algorithm"""

import numpy as np
import gym

# V(St) = V(St) + alpha * (Gt - V(St))
# Episode = [[states], [rewards]]


def generate_episode(env, policy, max_steps):
    """makes an episode of frozen lake"""
    episode = [[], []]
    state = env.reset()

    for t in range(max_steps):
        action = policy(state)
        new_state, reward, done, info = env.step(action)

        episode[0].append(state)
        if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
            episode[1].append(-1)
            return episode
        if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
            episode[1].append(1)
            return episode

        episode[1].append(0)
        state = new_state
    return episode


def monte_carlo(env,
                V,
                policy,
                episodes=5000,
                max_steps=100,
                alpha=0.1,
                gamma=0.99):
    """The monte carlo algorithm"""
    n = env.observation_space.n
    discounts = [gamma ** i for i in range(max_steps)]
    for ep in range(episodes):
        episode = generate_episode(env, policy, max_steps)

        # compute gt, update the value
        for i in range(len(episode[0])):
            Gt = np.sum(np.array(episode[1][i:])
                        * np.array(discounts[:len(episode[1][i:])]))
            V[episode[0][i]] = (V[episode[0][i]] + alpha
                                * (Gt - V[episode[0][i]]))

    return V
