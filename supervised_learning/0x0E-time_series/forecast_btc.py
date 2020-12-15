#!/usr/bin/env python3
""""""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def forcast():
    tf.enable_eager_execution()

    train_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22_preprocessed.csv"
    train_target_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22targets_preprocessed.csv"

    values = np.genfromtxt(train_path, delimiter=",", skip_header=True)
    targets = np.genfromtxt(train_target_path, delimiter=",", skip_header=True)

    values = values.reshape(1, 24, 1)
    targets = targets[np.newaxis].reshape(1, 1, 1)

    values = tf.data.Dataset.from_tensor_slices([values])
    values = values.repeat()
    targets = tf.data.Dataset.from_tensor_slices([targets])
    targets = targets.repeat()

    train_dataset = tf.data.Dataset.zip((values, targets))

    model = keras.models.Sequential([keras.layers.InputLayer(input_shape=(1, 24, 1)), keras.layers.LSTM(units=1, return_sequences=True)])

    model.compile(loss="mean_squared_error", optimizer=tf.train.AdamOptimizer())

    history = model.fit(train_dataset, steps_per_epoch=1, validation_split=False, epochs=1)

    print(history.history)

    return history

history = forcast()
