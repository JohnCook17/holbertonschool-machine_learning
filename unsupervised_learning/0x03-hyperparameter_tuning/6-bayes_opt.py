#!/usr/bin/env python3
"""Transfer learning with xception"""
import tensorflow.keras as K
from GPyOpt.methods import BayesianOptimization
import pickle
import os


class my_model():
    """"""

    def make_model(self, param):
        """"""
        self.lr = param[0][0]
        dr = param[0][1]
        layer_units0 = param[0][2]
        layer_units1 = param[0][3]
        layer_units2 = param[0][4]

        def learning_rate(epoch):
            """The learning rate scheduler"""
            return self.lr

        """Do not touch from here..."""
        # load data
        (X, Y), (X_test, Y_test) = K.datasets.cifar10.load_data()
        # uncomment for rapid test
        # X = X[0:256, :, :, :]
        # Y = Y[0:256, :]
        # X_test = X_test[0:256, :, :, :]
        # Y_test = Y_test[0:256, :]
        # preprocessing
        Y = K.utils.to_categorical(Y[:])
        X = K.applications.xception.preprocess_input(X)
        Y_test = K.utils.to_categorical(Y_test[:])
        X_test = K.applications.xception.preprocess_input(X_test)
        # data format
        df = "channels_last"
        # call backs
        save_best = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                                monitor="val_acc",
                                                save_best_only=True,
                                                )
        early_stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=10
                                               )
        learning_rate_0 = K.callbacks.LearningRateScheduler(learning_rate,
                                                            verbose=1
                                                            )
        # input layer and lambda layer save and load for faster training
        try:
            loaded_model = K.models.load_model("frozen_layers.h5")
            print("Loaded frozen layers!")
        except Exception as e:
            if isinstance(e, OSError):
                pass
            else:
                exit()
            print("Failed to load frozen layers.")
            inputs = K.Input(shape=(32, 32, 3))
            l = K.layers.Lambda(lambda X:
                                K.backend.resize_images(X,
                                                        height_factor=7,
                                                        width_factor=7,
                                                        data_format="channels_last"
                                                        ))(inputs)
            # Transfer learning layers
            xception = K.applications.Xception(include_top=False,
                                               input_tensor=l,
                                               weights="imagenet",
                                               pooling="max"
                                               )
            # freeze the resnet50 layers
            for layer in xception.layers:
                layer.trainable = False
            # get outputs
            outputs = xception.layers[-1].output
            outputs = K.layers.Dense(units=10,
                                     activation="softmax",
                                     kernel_initializer=K.initializers.he_normal()
                                     )(outputs)
            # compile frozen model
            model = K.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy"])
            model.fit(X,
                      Y,
                      epochs=1,
                      verbose=True,
                      batch_size=128
                      )
            model.save("frozen_layers.h5")
            loaded_model = K.models.load_model("frozen_layers.h5")
        except MemoryError("Try lowering the batch size"):
            exit()
        # set up new model
        if os.path.exists("X_inputs") and os.path.exists("X_test_inputs"):
            with open("X_inputs", "rb") as X_file:
                X = X_file
            with open("X_test_inputs", "rb") as X_test_file:
                X_test = X_test_file
        else:
            frozen_layers = K.Model(inputs=loaded_model.input,
                                    outputs=loaded_model.layers[-2].output
                                    )
            X = frozen_layers.predict(X,
                                      verbose=True
                                      )
            X_test = frozen_layers.predict(X_test,
                                           verbose=True
                                           )
            with open("X_inputs", "wb") as X_file:
                pickle.dump(X, X_file)
            with open("X_test_inputs", "wb") as X_test_file:
                pickle.dump(X_test, X_test_file)

        # inputs
        inputs = K.Input((2048,))
        """... to here!!!"""
        # new layers here
        layer = K.layers.Dense(units=layer_units0,
                               activation="relu",
                               kernel_initializer=K.initializers.he_normal()
                               )(inputs)
        layer = K.layers.Dropout(dr)(layer)
        layer = K.layers.Dense(units=layer_units1,
                               activation="relu",
                               kernel_initializer=K.initializers.he_normal()
                               )(layer)
        layer = K.layers.Dropout(dr)(layer)
        layer = K.layers.Dense(units=layer_units2,
                               activation="relu",
                               kernel_initializer=K.initializers.he_normal()
                               )(layer)
        layer = K.layers.Dropout(dr)(layer)
        outputs = K.layers.Dense(units=10,
                                 activation="softmax",
                                 kernel_initializer=K.initializers.he_normal()
                                 )(layer)
        model = K.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        # train
        h = model.fit(X,
                      Y,
                      validation_data=(X_test, Y_test),
                      epochs=64,
                      verbose=True,
                      batch_size=128,
                      shuffle=True,
                      callbacks=[early_stop, learning_rate_0, save_best]
                      )

        val_accuracy = np.max(h.history["val_acc"])

        return val_accuracy

    def opt(self):
        """"""
        search_space = [
            {"name": "lr", "type": "continuous", "domain": (0.01, 0.001)},
            {"name": "dr", "type": "continuous", "domain": (0.1, 0.3)},
            {"name": "layer_units0", "type": "discrete", "domain": (32, 64, 128, 256)},
            {"name": "layer_units1", "type": "discrete", "domain": (32, 64, 128, 256)},
            {"name": "layer_units2", "type": "discrete", "domain": (32, 64, 128, 256)}
        ]
        bayesian_opt = BayesianOptimization(self.make_model,
                                            domain=search_space,
                                            model_type="GP",
                                            initial_design_type="random",
                                            acquisition_type="EI",
                                            batch_size=1,
                                            maximize=True,
                                            verbosity=True
                                            )


def preprocess_data(X, Y):
    """The data preprocessing"""
    Y_p = K.utils.to_categorical(Y[:])
    X_p = K.applications.xception.preprocess_input(X)
    loaded_model = K.models.load_model("frozen_layers.h5")
    frozen_layers = K.Model(inputs=loaded_model.input,
                            outputs=loaded_model.layers[-2].output
                            )
    X_p = frozen_layers.predict(X_p,
                                verbose=True
                                )
    with open("Preprocessed_data_Xs", "wb") as my_file0:
        pickle.dump(X_p, my_file0)
    with open("Preprocessed_data_Ys", "wb") as my_file1:
        pickle.dump(Y_p, my_file1)
    return X_p, Y_p
