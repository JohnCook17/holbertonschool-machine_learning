#!/usr/bin/env python3
""""""
import tensorflow.keras as K
model = __import__('6-transfer').MyModel

# fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase
"""
_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
"""

run = model()
run.my_model()
