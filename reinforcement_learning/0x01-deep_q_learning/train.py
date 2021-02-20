#!/usr/bin/env python3
"""Trains the module for playing breakout using the EpsGreedyQPolicy()
   for training and the GreedyQPolicy for testing."""
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

ENV_NAME = 'BreakoutDeterministic-v4'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
weights_filename = 'policy.h5'


class AtariProcessor(Processor):
    """The environment in which the game will be played"""
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
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


def create_model(weights_filename, input_shape):
    """The CNN model to be used by the deep q learning model."""
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
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.,
                              value_min=.1,
                              value_test=.05,
                              nb_steps=1000000)
dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               test_policy=GreedyQPolicy(),
               memory=memory,
               processor=processor,
               nb_steps_warmup=50000,
               gamma=.95,
               target_model_update=10000,
               train_interval=4,
               delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.epsilon = 100

checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5'
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)


checkpoint_weights_filename = 'policy_{step}.h5'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                     interval=100000)]
callbacks += [FileLogger(log_filename, interval=100000)]


dqn.fit(env,
        callbacks=callbacks,
        nb_steps=5000000,
        log_interval=50000,
        visualize=True,
        verbose=1)
dqn.save_weights('policy.h5'.format(ENV_NAME), overwrite=True)
