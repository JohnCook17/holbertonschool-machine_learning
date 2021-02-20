#!/usr/bin/env python3
""""""
from __future__ import division
import numpy as np
import gym
from gym import wrappers
import os.path
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.policy import (LinearAnnealedPolicy,
                       BoltzmannQPolicy,
                       EpsGreedyQPolicy,
                       GreedyQPolicy)
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = 'BreakoutDeterministic-v0'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
weights_filename = 'policy_4500000.h5'


class AtariProcessor(Processor):
    """"""
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, './breakout', force=True)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


def create_model(weights_filename, input_shape):
    weights_filename = "./" + weights_filename
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


model = create_model(weights_filename, input_shape)
print(model.summary())


memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

processor = AtariProcessor()
policy = GreedyQPolicy()
dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=50000,
               gamma=.95,
               target_model_update=10000,
               train_interval=4,
               delta_clip=1.)

dqn.epsilon = 100

dqn.compile(Adam(lr=.00025), metrics=['mae'])

if os.path.isfile(weights_filename):
    print('\n\n\n\nSaved parameters found. I will use this file...\n'
          + weights_filename + '\n\n\n\n')
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
else:
    ('\n\n\n\nSaved parameters Not found...\n\n\n\n')
