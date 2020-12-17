#!/usr/bin/env python3
""""""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path


def forcast():
    tf.enable_eager_execution()

    train_path = "data/bitstampUSD_preprocessed.csv"
    train_target_path = "data/bitstampUSD_targets_preprocessed.csv"

    validate_path = "data/coinbaseUSD_preprocessed.csv"
    validate_target_path = "data/coinbaseUSD_targets_preprocessed.csv"

    values = np.genfromtxt(train_path,
                           delimiter=",",
                           skip_header=True,
                           dtype="float32")

    targets = np.genfromtxt(train_target_path,
                            delimiter=",",
                            skip_header=True,
                            dtype="float32")

    validate_vs = np.genfromtxt(validate_path,
                                delimiter=",",
                                skip_header=True,
                                dtype="float32")

    validate_ts = np.genfromtxt(validate_target_path,
                                delimiter=",",
                                skip_header=True,
                                dtype="float32")

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

    validation_dataset = tf.data.Dataset.zip((validate_values,
                                              validate_target))

    model = (tf.keras.models
             .Sequential([tf.keras.layers.InputLayer(input_shape=(1, 24, 1)),
                          tf.keras.layers.LSTM(units=1),
                          tf.keras.layers.Dense(units=1)]))

    model.compile(loss="mean_squared_error",
                  optimizer=tf.train.AdamOptimizer(learning_rate=10))  # 1

    history = model.fit(train_dataset,
                        steps_per_epoch=1,
                        batch_size=1,
                        validation_split=False,
                        epochs=100,  # 4500 is ideal for val set
                        validation_data=validation_dataset)

    model.save_weights("saved_model/lstm.h5")

    plt.plot(history.history["loss"], "b")
    plt.plot(history.history["val_loss"], "r")
    plt.show()

    return model


def prediction(prediction_path, model, reshape_shape):

    prediction_data = np.genfromtxt(prediction_path,
                                    delimiter=",",
                                    skip_header=True)
    print(prediction_data.shape)
    prediction_data = prediction_data[np.newaxis].reshape(reshape_shape)
    print(prediction_data.shape)
    # prediction_data = tf.data.Dataset.from_tensor_slices([prediction_data])
    # prediction_data = prediction_data.repeat()
    prediction_data = (prediction_data,)
    # prediction_data = prediction_data.tolist()
    print(prediction_data)

    my_prediction = model.predict(prediction_data, steps=1)

    print("my pred = ", my_prediction)
    return my_prediction.flatten()


if __name__ == "__main__":
    if path.exists("saved_model/lstm.h5"):
        model = (tf.keras.models
                 .Sequential([tf.keras.layers.
                              InputLayer(input_shape=(24, 1)),
                              tf.keras.layers.LSTM(units=1),
                              tf.keras.layers.Dense(units=1)]))
        model.load_weights("saved_model/lstm.h5")
        model.compile(loss="mean_squared_error",
                      optimizer=tf.train.AdamOptimizer(learning_rate=1))
        reshape_shape = (1, 24, 1)
    else:
        model = forcast()
        reshape_shape = (24, 1, 1, 1)

    my_datasets = ["coinbase", "bitstamp"]

    for dataset in my_datasets:
        predictions = []
        for i in range(7):
            prediction_path = "data/{}USDday{}.csv".format(dataset, i)
            predictions.append(prediction(prediction_path,
                                          model,
                                          reshape_shape))

        predictions = np.asarray(predictions)
        targets = np.genfromtxt("data/{}USD_7_day_targets.csv".format(dataset),
                                delimiter=",",
                                skip_header=True)

        targets = targets[np.newaxis].reshape(7, 1)

        print(predictions.shape)
        print(targets.shape)

        plt.plot(predictions, "b")
        plt.plot(targets, "r")
        plt.show()
