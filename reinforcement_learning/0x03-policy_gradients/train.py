#!/usr/bin/env python3
"""Trains the model using policy gradient"""
import gym
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    W = np.random.random((4, 2))  # number of states and actions

    episode_rewards = []

    for e in range(nb_episodes):
        state = env.reset()[None, :]

        grads = []
        rewards = []

        score = 0
        while True:
            if show_result and (e / 1000 == 0):
                env.render

            action, grad = policy_gradient(state, W)
            next_state, reward, done, info = env.step(action)
            next_state = next_state[None, :]
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

            if done:
                break

        for i in range(len(grads)):
            W += alpha * grads[i] * sum([r * gamma ** r
                                         for t, r in enumerate(rewards[i:])])

        episode_rewards.append(score)
        print("{}: {}".format(e, score), end="\r", flush=False)

    return episode_rewards
