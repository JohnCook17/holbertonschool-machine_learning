#!/usr/bin/env python3
""""""
import tensorflow.keras as K

if __name__ == "__main__":

    def learning_rate(epoch):
        """"""
        return 0.001 / (1 + 1 * epoch)

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
                                           patience=3
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
        lam = K.layers.Lambda(lambda X:
                              K.backend.resize_images(X,
                                                      height_factor=7,
                                                      width_factor=7,
                                                      data_format=df
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
    except MemoryError("Try lowering the batch size"):
        exit()
    # set up new model
    frozen_layers = K.Model(inputs=loaded_model.input,
                            outputs=loaded_model.layers[-2].output
                            )
    X = frozen_layers.predict(X,
                              verbose=True
                              )
    X_test = frozen_layers.predict(X_test,
                                   verbose=True
                                   )
    # inputs
    inputs = K.Input((2048,))
    # new layers here
    layer = K.layers.Dense(units=512,
                           activation="relu",
                           kernel_initializer=K.initializers.he_normal()
                           )(inputs)
    # D_0 = K.layers.Dropout(0.5)(FC_0)
    layer = K.layers.Dense(units=256,
                           activation="relu",
                           kernel_initializer=K.initializers.he_normal()
                           )(layer)
    # D_1 = K.layers.Dropout(0.5)(FC_1)
    layer = K.layers.Dense(units=128,
                           activation="relu",
                           kernel_initializer=K.initializers.he_normal()
                           )(layer)
    # model.add(K.layers.Dropout(0.5))
    outputs = K.layers.Dense(units=10,
                             activation="softmax",
                             kernel_initializer=K.initializers.he_normal()
                             )(layer)
    model = K.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
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


if __name__ != "__main__":
    def preprocess_data(X, Y):
        """"""
        Y_p = K.utils.to_categorical(Y[:])
        X_p = K.applications.xception.preprocess_input(X)
        return X_p, Y_p
