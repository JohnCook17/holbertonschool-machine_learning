#!/usr/bin/env python3
""""""
import tensorflow as tf


def lenet5(x, y):
    """"""
    m = tf.shape(y)[0]
    classes = y.shape[1]
    init_method = tf.contrib.layers.variance_scaling_initializer()
    weights = {
        "cnnf1": tf.Variable(init_method(shape=(5, 5, 1, 6))),
        "cnnf2": tf.Variable(init_method(shape=(5, 5, 6, 16))),
        "fullc1": tf.Variable(init_method(shape=(5 * 5 * 16, 120))),
        "fullc2": tf.Variable(init_method(shape=(120, 84))),
        "output": tf.Variable(init_method(shape=(84, int(classes))))
    }
    biases = {
        "cnnf1": tf.Variable(tf.zeros(6)),
        "cnnf2": tf.Variable(tf.zeros(16)),
        "fullc1": tf.Variable(tf.zeros(120)),
        "fullc2": tf.Variable(tf.zeros(84)),
        "output": tf.Variable(tf.zeros(classes))
    }
    layer1 = tf.nn.conv2d(input=x, filter=weights["cnnf1"], padding="SAME", strides=[1, 1, 1, 1])
    layer1 = tf.nn.bias_add(layer1, biases["cnnf1"])
    layer1 = tf.nn.relu(layer1)
    p1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    layer2 = tf.nn.conv2d(input=p1, filter=weights["cnnf2"], padding="VALID", strides=[1, 1, 1, 1])
    layer2 = tf.nn.bias_add(layer2, biases["cnnf2"])
    layer2 = tf.nn.relu(layer2)
    p2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    p2 = tf.reshape(p2, [m, 400])
    layer3 = tf.add(tf.matmul(p2, weights["fullc1"]), biases["fullc1"])
    layer3 = tf.nn.relu(layer3)
    layer4 = tf.add(tf.matmul(layer3, weights["fullc2"]), biases["fullc2"])
    layer4 = tf.nn.relu(layer4)
    layer5 = tf.add(tf.matmul(layer4, weights["output"]), biases["output"])
    logits = tf.nn.softmax(layer5)
    y_label = tf.argmax(y, 1)
    pred = tf.argmax(logits, 1)
    equal = tf.equal(pred, y_label)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y)
    train = tf.train.AdamOptimizer().minimize(loss)
    return logits, train, loss, accuracy
