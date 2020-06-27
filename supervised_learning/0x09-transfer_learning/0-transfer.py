#!/usr/bin/env python3
""""""
import tensorflow.keras as K

if __name__ == "__main__":

    def learning_rate(epoch):
        """"""
        return 0.001 / (1 + 1 * epoch)

    # load data
    (X, Y), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X = X[0:256, :, :, :]
    Y = Y[0:256, :]
    # preprocessing
    Y = K.utils.to_categorical(Y[:])
    X = K.applications.xception.preprocess_input(X)
    Y_test = K.utils.to_categorical(Y_test[:])
    X_test = K.applications.xception.preprocess_input(X_test)
    # call backs
    save_best = K.callbacks.ModelCheckpoint(filepath="cifar10.h5",
                                            monitor="val_acc",
                                            save_best_only=True,
                                            )
    early_stop = K.callbacks.EarlyStopping(monitor="loss",
                                           patience=3
                                           )
    learning_rate_0 = K.callbacks.LearningRateScheduler(learning_rate,
                                                        verbose=1
                                                        )
    # input layer and lambda layer save and load for faster training
    try:
        loaded_model = K.models.load_model("frozen_layers.h5")
        inputs = loaded_model.layers[-2].output
    except Exception as e:
        if isinstance(e, OSError):
            pass
        else:
            exit()
        inputs = K.Input(shape=(32, 32, 3))
        lam = K.layers.Lambda(lambda X:
                              K.backend.resize_images(X,
                                                      height_factor=9,
                                                      width_factor=9,
                                                      data_format="channels_last"
                                                      ))(inputs)
        # Transfer learning layers
        xception = K.applications.Xception(include_top=False,
                                           input_tensor=lam,
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
                      metrics=["acc"])
        model.fit(X,
                  Y,
                  epochs=1,
                  verbose=True,
                  batch_size=128
                  )
        model.save("frozen_layers.h5")
        loaded_model = K.models.load_model("frozen_layers.h5")
        inputs = loaded_model.layers[-2].output
    except MemoryError("Try lowering the batch size"):
        exit()
    # new layers here
    FC_0 = K.layers.Dense(units=512,
                          activation="relu",
                          kernel_initializer=K.initializers.he_normal()
                          )(inputs)
    # D_0 = K.layers.Dropout(0.5)(FC_0)
    FC_1 = K.layers.Dense(units=256,
                          activation="relu",
                          kernel_initializer=K.initializers.he_normal()
                          )(FC_0)
    # D_1 = K.layers.Dropout(0.5)(FC_1)
    FC_2 = K.layers.Dense(units=128,
                          activation="relu",
                          kernel_initializer=K.initializers.he_normal()
                          )(FC_1)
    D_2 = K.layers.Dropout(0.5)(FC_2)
    outputs = K.layers.Dense(units=10,
                             activation="softmax",
                             kernel_initializer=K.initializers.he_normal()
                             )(D_2)
    model = K.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    # train
    model.fit(X,
              Y,
              validation_data=(X_test, Y_test),
              epochs=64,
              verbose=True,
              batch_size=128,
              shuffle=True,
              callbacks=[early_stop, learning_rate_0, save_best]
              )
    model.save("cifar10.h5")


if __name__ != "__main__":
    def preprocess_data(X, Y):
        """"""
        Y_p = K.utils.to_categorical(Y[:])
        X_p = K.applications.xception.preprocess_input(X)
        return X_p, Y_p
