#!/usr/bin/env python3
""""""
from tensorflow import keras


class TripletLoss(keras.layers.Layer):
    """"""
    def __init__(self, alpha, **kwargs):
        """"""
        super().__init__(**kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """"""
        anchor, positive, negative = inputs
        term1 = keras.backend.sum(keras.backend.square(anchor - positive), axis=-1)
        term2 = keras.backend.sum(keras.backend.square(anchor - negative), axis=-1)
        return keras.backend.maximum(term1 - term2 + self.alpha, 0)

    def call(self, inputs):
        """"""
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

    def compute_output_shape(self, inputs):
        """"""
        return 128
