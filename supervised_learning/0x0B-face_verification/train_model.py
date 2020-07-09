#!/usr/bin/env python3
""""""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from triplet_loss import TripletLoss


class TrainModel():
    """"""
    def __init__(self, model_path, alpha):
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)
            # self.base_model.summary()
            # print(self.base_model.inputs[0].shape)
            shape = self.base_model.inputs[0].shape[1:]
            a_input = tf.keras.Input(shape=shape, name="a_input")
            p_input = tf.keras.Input(shape=shape, name="p_input")
            n_input = tf.keras.Input(shape=shape, name="n_input")
            encoded_a = self.base_model(a_input)
            encoded_p = self.base_model(p_input)
            encoded_n = self.base_model(n_input)
            encoded_inputs = [encoded_a, encoded_p, encoded_n]
            loss_layer = TripletLoss(alpha=alpha)(encoded_inputs)
            train_model = tf.keras.Model(inputs=[a_input, p_input, n_input], outputs=loss_layer)
            opt = tf.keras.optimizers.Adam()
            train_model.compile(optimizer=opt)
            self.training_model = train_model

    def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3, verbose=True):
        """"""
        # trim off last batch
        train_data = triplets[0].shape[0]
        while(train_data * (1 - validation_split)) % batch_size != 0 and (train_data * validation_split) % batch_size != 0:
            train_data -= 1
            # print(train_data)
        # train
        trimed_triplets = [triplets[0][:train_data], triplets[1][:train_data], triplets[2][:train_data]]
        return self.training_model.fit(trimed_triplets,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=verbose,
                                       validation_split=validation_split
                                       )

    def save(self, save_path):
        """"""
        self.base_model.save(save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """"""
        with tf.Session() as sess:
            y_true = K.constant(y_true)
            y_pred = K.constant(y_pred)
            # calculate recall
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            # calculate precision
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            # calculate f1
            f1_m = 2*((precision*recall)/(precision+recall+K.epsilon()))
            init = tf.initialize_all_variables()
            sess.run(init)
            return f1_m.eval()

    @staticmethod
    def accuracy(y_true, y_pred):
        """"""
        with tf.Session() as sess:
            acc = K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))).eval()
            init = tf.initialize_all_variables()
            sess.run(init)
            return acc
