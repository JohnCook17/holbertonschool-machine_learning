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


    print(values.shape)
    print(targets.shape)


    values = values.reshape(24, 1)
    targets = targets.reshape(1, 1)

    values = tf.data.Dataset.from_tensor_slices([values])
    targets = tf.data.Dataset.from_tensor_slices([targets])

    train_dataset = tf.data.Dataset.zip((values, targets))

    model = keras.models.Sequential([keras.layers.LSTM(units=(24, 1), return_sequences=True)])

    model.compile(loss="mean_squared_error", optimizer=tf.train.AdamOptimizer())

    print(train_dataset, "\n")
    for i, element in enumerate(train_dataset.take(10)):
        print(i)
        print(element)
        

    history = model.fit(train_dataset, steps_per_epoch=1, validation_split=False, epochs=1)

    return history

history = forcast()
print(history)
