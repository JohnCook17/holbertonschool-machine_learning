#!/usr/bin/env python3
""""""
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the frozen lake"""
    env = gym.make("FrozenLake-v0", desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)
    return env


def epsilon_greedy(Q, state, epsilon):
    """always exploit q table"""
    env = load_frozen_lake()
    # print(env)
    action = np.argmax(Q[state, :])
    return action


def train(env,
          Q,
          episodes=1,
          max_steps=100,
          alpha=0.1,
          gamma=0.99,
          epsilon=1,
          min_epsilon=0.1,
          epsilon_decay=0.05):
    """Trains the 'game' one episode"""
    rewards = []
    for _ in range(episodes):
        state = env.reset()

        if epsilon < min_epsilon:
            epsilon = min_epsilon

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            Q[state][action] += epsilon_decay * (reward + gamma *
                                                 np.max(Q[new_state]) -
                                                 Q[state][action])

            state = new_state

            env.render()

            if done:
                break

        rewards.append(reward)

    return Q, rewards


def play(env, Q, max_steps=100):
    """plays the 'game' and returns the total reward"""
    Q, rewards = train(env, Q, max_steps=max_steps)

    return np.max(rewards)
