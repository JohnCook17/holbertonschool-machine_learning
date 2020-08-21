#!/usr/bin/env python3
"""Transfer learning with xception"""
import GPyOpt
import os
import pickle as pkl
import tensorflow.keras as K


if __name__ != "__main__":

    def train(lr,
              dr,
              batch_size,
              fc_size0,
              fc_size1,
              fc_size2,
              toy):
        """"""

        def learning_rate(epoch):
            """The learning rate scheduler"""
            return lr / (1 + 1 * epoch)

        # load data
        (X, Y), (X_test, Y_test) = K.datasets.cifar10.load_data()
        # uncomment for rapid test
        if toy is True:
            X = X[0:256, :, :, :]
            Y = Y[0:256, :]
            X_test = X_test[0:256, :, :, :]
            Y_test = Y_test[0:256, :]
        # preprocessing
        Y = K.utils.to_categorical(Y[:])
        X = K.applications.xception.preprocess_input(X)
        Y_test = K.utils.to_categorical(Y_test[:])
        X_test = K.applications.xception.preprocess_input(X_test)
        # data format
        df = "channels_last"
        # call backs
        save_best = K.callbacks.ModelCheckpoint(filepath="model_checkpoint_" +
                                                + "lr" + str(lr) + "_" +
                                                + "dr" + str(dr) + "_" +
                                                + "fc_size0" +
                                                str(fc_size0) + "_" +
                                                + "fc_size1" +
                                                str(fc_size1) + "_" +
                                                + "fc_size2" +
                                                str(fc_size2) + "_" +
                                                ".hdf5",
                                                monitor="val_loss",
                                                verbose=1,
                                                save_best_only=True)
        early_stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=5
                                               )
        learning_rate_0 = K.callbacks.LearningRateScheduler(learning_rate,
                                                            verbose=1
                                                            )
        # input layer and lambda layer save and load for faster training
        try:
            loaded_model = K.models.load_model("frozen_layers.h5")
        except Exception as e:
            if isinstance(e, OSError):
                pass
            else:
                exit()
            inputs = K.Input(shape=(32, 32, 3))
            ilayer = K.layers.Lambda(lambda X:
                                     K.backend.resize_images(X,
                                                             height_factor=7,
                                                             width_factor=7,
                                                             data_format=df
                                                             ))(inputs)
            # Transfer learning layers
            xception = K.applications.Xception(include_top=False,
                                               input_tensor=ilayer,
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
                                     kernel_initializer=K.
                                     initializers.he_normal()
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
                      batch_size=128  # keep this at 128 as a nice middle.
                      )
            model.save("frozen_layers.h5")
            loaded_model = K.models.load_model("frozen_layers.h5")
        except MemoryError("Try lowering the batch size"):
            exit()
        # set up new model, pickle np arrays
        frozen_layers = K.Model(inputs=loaded_model.input,
                                outputs=loaded_model.layers[-2].output
                                )

        # Create the inputs X
        if not os.path.exists("Preprocessed_X.pkl"):
            X = frozen_layers.predict(X,
                                      verbose=True
                                      )
            with open("Preprocessed_X.pkl", "wb") as f:
                pkl.dump(X, f)

        # Create the test X
        if not os.path.exists("Preprocessed_X_test.pkl"):
            X_test = frozen_layers.predict(X_test,
                                           verbose=True
                                           )
            with open("Preprocessed_X_test.pkl", "wb") as f1:
                pkl.dump(X_test, f1)

        # Load the inputs X
        with open("Preprocessed_X.pkl", "rb") as f2:
            X = pkl.load(f2)
        # Load the imputs test X
        with open("Preprocessed_X_test.pkl", "rb") as f3:
            X_test = pkl.load(f3)
        # inputs
        inputs = K.Input((2048,))
        # new layers here
        layer = K.layers.Dense(units=fc_size0,
                               activation="relu",
                               kernel_initializer=K.initializers.he_normal()
                               )(inputs)
        layer = K.layers.Dropout(dr)(layer)
        layer = K.layers.Dense(units=fc_size1,
                               activation="relu",
                               kernel_initializer=K.initializers.he_normal()
                               )(layer)
        layer = K.layers.Dropout(dr)(layer)
        layer = K.layers.Dense(units=fc_size2,
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
        # print("================================")
        history = model.fit(X,
                            Y,
                            validation_data=(X_test, Y_test),
                            epochs=100,
                            verbose=True,
                            batch_size=batch_size,
                            shuffle=True,
                            callbacks=[early_stop, learning_rate_0, save_best]
                            )
        evaluation = model.evaluate(X_test,
                                    Y_test,
                                    batch_size=batch_size,
                                    verbose=1
                                    )
        return evaluation  # might need to change to history

    def preprocess_data(self, X, Y):
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
        return X_p, Y_p


class MyModel():
    """"""
    def __init__(self):
        """"""
        # set patience to same, epochs too
        self.bounds = [{"name": "lr", "type": "continuous", "domain":
                        (.001, .0001)},
                       {"name": "dr", "type": "continuous", "domain":
                        (.1, .9)},
                       {"name": "batch_size", "type": "discrete", "domain":
                        (32, 64, 128, 256)},
                       {"name": "fc_size0", "type": "discrete", "domain":
                        (32, 64, 128, 256, 512, 1024)},
                       {"name": "fc_size1", "type": "discrete", "domain":
                        (32, 64, 128, 256, 512, 1024)},
                       {"name": "fc_size2", "type": "discrete", "domain":
                        (32, 64, 128, 256, 512, 1024)}
                       ]
        self.train = train

    def unpacker(self, x):
        """"""
        evaluation = self.train(lr=float(x[:, 0]),
                                dr=float(x[:, 1]),
                                batch_size=int(x[:, 2]),
                                fc_size0=int(x[:, 3]),
                                fc_size1=int(x[:, 4]),
                                fc_size2=int(x[:, 5]),
                                toy=True)  # set to False to train real model
        print("\n\n======================")
        print(evaluation)
        print("======================\n\n")
        return evaluation[0]

    def my_model(self):
        """"""
        opt_hyper_p = GPyOpt.methods.BayesianOptimization(f=self.unpacker,
                                                          domain=self.bounds,
                                                          maximize=False,
                                                          verbosity=True)
        opt_hyper_p.run_optimization(max_iter=1,
                                     verbosity=True,
                                     report_file="bayes_opt.txt")
        opt_hyper_p.plot_convergence()
