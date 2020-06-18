#!/usr/bin/env python3
"""resnet50 in tensorflow keras"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """resnet50 in tensorflow keras"""
    initializer = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    conv2d = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=(2, 2),
                             padding="same",
                             kernel_initializer=initializer)(inputs)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation("relu")(batch_normalization)
    max_pooling2d = K.layers.MaxPooling2D(pool_size=(3, 3),
                                          strides=(2, 2),
                                          padding="same")(activation)
    # conv2_x
    pro0 = projection_block(max_pooling2d, (64, 64, 256), 1)
    id0 = identity_block(pro0, (64, 64, 256))
    id1 = identity_block(id0, (64, 64, 256))
    # conv3_x
    pro1 = projection_block(id1, (128, 128, 512), 2)
    id2 = identity_block(pro1, (128, 128, 512))
    id3 = identity_block(id2, (128, 128, 512))
    id4 = identity_block(id3, (128, 128, 512))
    # conv4_x
    pro2 = projection_block(id4, (256, 256, 1024), 2)
    id5 = identity_block(pro2, (256, 256, 1024))
    id6 = identity_block(id5, (256, 256, 1024))
    id7 = identity_block(id6, (256, 256, 1024))
    id8 = identity_block(id7, (256, 256, 1024))
    id9 = identity_block(id8, (256, 256, 1024))
    # conv5_x
    pro3 = projection_block(id9, (512, 512, 2048), 2)
    id10 = identity_block(pro3, (512, 512, 2048))
    id11 = identity_block(id10, (512, 512, 2048))
    average_pooling2d = K.layers.AveragePooling2D(pool_size=(7, 7),
                                                  strides=(1, 1))(id11)
    softmax = K.layers.Dense(units=1000,
                             activation="softmax",
                             kernel_initializer=initializer)(average_pooling2d)
    model = K.Model(inputs=inputs, outputs=softmax)
    return model
