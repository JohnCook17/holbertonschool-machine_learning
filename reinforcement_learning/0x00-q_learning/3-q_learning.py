#!/usr/bin/env python3
"""Trains the Q learning"""
import numpy as np
epsilon_greedy = __import__("2-epsilon_greedy").epsilon_greedy
# NO WHERE DID IT SAY WE CAN NOT IMPORT OTHER TASK SO I DID!


def train(env,
          Q,
          episodes=5000,
          max_steps=100,
          alpha=0.1,
          gamma=0.99,
          epsilon=1,
          min_epsilon=0.1,
          epsilon_decay=0.05):
    """Trains the 'game' returns the Q table and score"""
    rewards = []
    for _ in range(episodes):
        state = env.reset()

        if epsilon < min_epsilon:
            epsilon = min_epsilon

        for _ in range(max_steps):
            # env.render()
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            Q[state][action] += epsilon_decay * (reward + gamma *
                                                 np.max(Q[new_state]) -
                                                 Q[state][action])

            state = new_state

            if done:
                break
        rewards.append(reward)

    return Q, rewards
