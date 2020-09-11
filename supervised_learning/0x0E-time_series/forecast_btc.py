#!/usr/bin/env python3
"""Keras prediction of btc price"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def make_data_split(look_back=60):
    my_data = np.genfromtxt("clean_data_t-1.csv", delimiter=",")
    my_data = my_data[np.newaxis]
    time_steps = my_data.shape[1]
    data_split = int(time_steps * 0.67)
    my_train_data = my_data  # [:, time_steps - data_split:, :]
    my_test_data = my_data  # [:, data_split:, :]
    my_train_data, time_stamps_train = my_train_data[:, :, 1:], my_train_data[:, :, 0, np.newaxis]
    my_test_data, time_stamps_test = my_test_data[:, :, 1:], my_test_data[:, :, 0, np.newaxis]
    my_train_y = my_train_data[:, -look_back:]
    my_train_data = my_train_data[:, :-look_back]
    my_train_data = tf.constant(my_train_data, shape=my_train_data.shape, dtype=tf.float32)
    my_train_y = tf.constant(my_train_y, shape=my_train_y.shape, dtype=tf.float32)
    my_test_data = tf.constant(my_test_data, shape=my_test_data.shape, dtype=tf.float32)
    return my_train_data, my_train_y, my_test_data, time_stamps_train, time_stamps_test


def forecast():
    """"""
    # get data set
    train, y, test, time_stamps_train, time_staps_test = make_data_split(60)
    data_set = tf.data.Dataset.from_tensor_slices((train, y)).batch(1)
    test_data_set = data_set  # tf.data.Dataset.from_tensor_slices((test)).batch(1)
    input_shape = train.shape
    # build model
    print("=================== making model ================================")
    print(train.shape, y.shape)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(4, input_shape=(input_shape), activation="relu"))
    model.add(keras.layers.Dense(4, activation="relu"))
    # model.add(keras.layers.Dense(input_shape[2]))
    # compile model
    model.compile(loss="mse", optimizer=tf.train.AdamOptimizer(learning_rate=0.05))
    
    model.fit(x=data_set.repeat(), steps_per_epoch=1, epochs=10, batch_size=1, verbose=True, shuffle=False)
    res = model.predict(data_set, steps=1, verbose=True)
    print(res)
    res = np.sum(res)
    print(res)


forecast()
