#!/usr/bin/env python3
""""""
import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
import matplotlib.pyplot as plt
from os import path


def forcast():
    tf.enable_eager_execution()

    train_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22_preprocessed.csv"
    train_target_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22targets_preprocessed.csv"

    validate_path = "data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09_preprocessed.csv"
    validate_target_path = "data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09targets_preprocessed.csv"

    values = np.genfromtxt(train_path, delimiter=",", skip_header=True, dtype="float32")
    targets = np.genfromtxt(train_target_path, delimiter=",", skip_header=True, dtype="float32")
    
    validate_vs = np.genfromtxt(validate_path, delimiter=",", skip_header=True, dtype="float32")
    validate_ts = np.genfromtxt(validate_target_path, delimiter=",", skip_header=True, dtype="float32")

    values = values.reshape(1, 24, 1)
    targets = targets[np.newaxis].reshape(1, 1, 1)
    
    validate_vs = validate_vs.reshape(1, 24, 1)
    validate_ts = validate_ts.reshape(1, 1, 1)

    values = tf.data.Dataset.from_tensor_slices([values])
    values = values.repeat()
    targets = tf.data.Dataset.from_tensor_slices([targets])
    targets = targets.repeat()

    validate_values = tf.data.Dataset.from_tensor_slices([validate_vs])
    validate_values = validate_values.repeat()
    validate_target = tf.data.Dataset.from_tensor_slices([validate_ts])
    validate_target = validate_target.repeat()

    train_dataset = tf.data.Dataset.zip((values, targets))

    validation_dataset = tf.data.Dataset.zip((validate_values, validate_target))

    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(1, 24, 1)),
                                     tf.keras.layers.LSTM(units=1, return_sequences=True, activation="relu"),
                                     tf.keras.layers.Dense(units=1)])

    model.compile(loss="mean_squared_error", optimizer=tf.train.AdamOptimizer(learning_rate=1))

    history = model.fit(train_dataset, steps_per_epoch=1, batch_size=1, validation_split=False, epochs=4500, validation_data=validation_dataset)

    model.save("saved_model/lstm")

    return model


if __name__ == "__main__":

    if path.exists("saved_model/lstm"):
        model = tf.keras.models.load_model("saved_model/lstm")
        model.compile(loss="mean_squared_error", optimizer=tf.train.AdamOptimizer(learning_rate=1))
    else:
        model = forcast()

    history = model.history

    prediction_path = "data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09_prediction.csv"
    prediction_data = np.genfromtxt(prediction_path, delimiter=",", skip_header=True)

    prediction_data = prediction_data[np.newaxis].reshape(24, 1, 1, 1)
    print(prediction_data.shape)
    # prediction_data = tf.data.Dataset.from_tensor_slices([prediction_data])
    # prediction_data = prediction_data.repeat()
    prediction_data = (prediction_data,)
    # prediction_data = prediction_data.tolist()
    print(prediction_data)

    my_prediction = model.predict(prediction_data, steps=1)
    
    print(my_prediction)
