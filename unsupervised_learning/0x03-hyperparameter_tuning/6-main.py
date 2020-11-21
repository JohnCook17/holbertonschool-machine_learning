#!/usr/bin/env python3

import os
import pickle
import tensorflow.keras as K
preprocess_data = __import__('6-bayes_opt').preprocess_data
GPBO = __import__("6-bayes_opt").my_model()

# fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase

_, (X, Y) = K.datasets.cifar10.load_data()
if ((os.path.exists("Preprocessed_data_Xs") and
     os.path.exists("Preprocessed_data_Ys"))):
    with open("Preprocessed_data_Xs", "rb") as my_file0:
        X_p = pickle.load(my_file0)
    with open("Preprocessed_data_Ys", "rb") as my_file1:
        Y_p = pickle.load(my_file1)
else:
    X_p, Y_p = preprocess_data(X, Y)
gpyopt_bo = GPBO.opt()
# model = K.models.load_model('cifar10.h5')
# model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
