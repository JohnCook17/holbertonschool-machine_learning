#!/usr/bin/env python3
""""""
import tensorflow.keras as K

if __name__ == "__main__":

    def learning_rate(epoch):
        """"""
        return 0.001 / (1 + 1 * epoch)

    learning_rate_0 = K.callbacks.LearningRateScheduler(learning_rate,
                                                        verbose=1)
    (X, Y), (X_test, Y_test) = K.datasets.cifar10.load_data()
    # preprocessing
    Y = K.utils.to_categorical(Y[:])
    X = K.applications.resnet50.preprocess_input(X)
    # input layer and lambda layer
    inputs = K.Input(shape=(32, 32, 3))
    """
    lam = K.layers.Lambda(lambda X:
                          K.backend.resize_images(X,
                                                  height_factor=7,
                                                  width_factor=7,
                                                  data_format="channels_last"
                                                  ))(inputs)
    """
    # Transfer learning layers
    ResNet50 = K.applications.ResNet50(include_top=False,
                                       input_tensor=inputs,
                                       weights="imagenet",
                                       pooling="avg"
                                       )
    # freeze the resnet50 layers
    for layer in ResNet50.layers:
        layer.trainable = False
    # new layers here
    ResNet50 = ResNet50.layers[-1].output
    FC_0 = K.layers.Dense(units=512,
                          activation="relu",
                          kernel_initializer=K.initializers.he_normal()
                          )(ResNet50)
    # model.add(K.layers.Dropout(0.5))
    FC_1 = K.layers.Dense(units=256,
                          activation="relu",
                          kernel_initializer=K.initializers.he_normal()
                          )(FC_0)
    # model.add(K.layers.Dropout(0.5))
    FC_2 = K.layers.Dense(units=128,
                          activation="relu",
                          kernel_initializer=K.initializers.he_normal()
                          )(FC_1)
    # model.add(K.layers.Dropout(0.5))
    outputs = K.layers.Dense(units=10,
                             activation="softmax",
                             kernel_initializer=K.initializers.he_normal()
                             )(FC_2)
    model = K.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    callback = K.callbacks.EarlyStopping(monitor="loss", patience=3)
    model.fit(X,
              Y,
              epochs=50,
              verbose=True,
              batch_size=128,
              shuffle=True,
              callbacks=[callback, learning_rate_0]
              )
    model.save("cifar10.h5")


if __name__ != "__main__":
    def preprocess_data(X, Y):
        """"""
        Y_p = K.utils.to_categorical(Y[:])
        X_p = K.applications.resnet50.preprocess_input(X)
        return X_p, Y_p
