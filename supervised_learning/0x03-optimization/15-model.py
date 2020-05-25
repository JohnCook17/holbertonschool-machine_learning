#!/usr/bin/env python3
"""Mini batch gradient descent with Adam and learning rate decay."""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """x is the input data placeholder,
    y is for the onehot lables for input data"""
    x = tf.placeholder(float, [None, nx], name="x")
    y = tf.placeholder(float, [None, classes], name="y")
    return x, y


def shuffle_data(X, Y):
    """X is the first np.ndarray to shuffle. Y is the second, both have a shape
    of (m, nx), and (m, ny) respectivly m being the number of data points
    and the n value being the number of features in the respective letter
    returns shuffled x and y"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def calculate_accuracy(y, y_pred):
    """uses tf.reduce_mean, y is the lables, y_pred is the predictions"""
    y_label = tf.argmax(y, 1)
    pred = tf.argmax(y_pred, 1)
    equal = tf.equal(pred, y_label)
    return tf.reduce_mean(tf.cast(equal, tf.float32))


def calculate_loss(y, y_pred):
    """uses softmax to calculate loss, with y lables and y_pred
    predictions"""
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)


def create_layer(prev, n, activation):
    """prev is the previous layer, n is the number of nodes,
    activation is the activation, i is for initialzation, which
    uses he et al"""
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    if activation:
        layer = tf.layers.Dense(units=n, activation=None, kernel_initializer=i)
        layer = layer(prev)
        return activation(layer)
    else:
        layer = tf.layers.Dense(units=n, activation=None, kernel_initializer=i)
        layer = layer(prev)
        return layer


def create_batch_norm_layer(prev, n, activation):
    """prev is the previous layer, n is the shape of the new layer,
    and activation is the activation"""
    i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None, kernel_initializer=i)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    norm = tf.nn.batch_normalization(layer(prev), mean=mean, variance=variance,
                                     offset=beta, scale=gamma,
                                     variance_epsilon=1e-8)
    return activation(norm)


def forward_prop(prev, layer, activation):
    """prev is the previous layer, n is the shapes of the layer,
    activation is the activation to use on the layer, first_layer
    is a flag used to determine if the first layer needs to be
    normalized."""
    first_layer = 1
    for shape, act in zip(layer, activation):
        if act is None or first_layer:
            first_layer = 0
            prev = create_layer(prev, shape, act)
        else:
            layer = create_batch_norm_layer(prev, shape, act)
    return prev


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """loss is a loss tensor. alpha is the learning rate.
    beta1 is the weight used for the first moment. beta2
    is the weight used for the second moment. epsilon avoids
    zero division."""
    return tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                  beta2=beta2, epsilon=epsilon).minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """alpha the original learning rate. decay_rate the rate alpha decays at.
    global_step, how many times gradient descent has been run. decay_step
    the number of steps before alpha should decay again."""
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """"""
    with tf.Session() as sess:
        x, y = create_placeholders(Data_train[0].shape[1],
                                   Data_train[1].shape[1])
        y_pred = forward_prop(x, layers, activations)
        epoch = 0
        accuracy = calculate_accuracy(y, y_pred)
        loss = calculate_loss(y, y_pred)
        train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
        step = 1
        global_step = 0
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        while epoch <= epochs:
            xt, yt = Data_train
            xv, yv = Data_valid
            feed_dict_t = {x: xt, y: yt}
            feed_dict_v = {x: xv, y: yv}
            m = xt.shape[0]
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(sess.run(loss, feed_dict_t)))
            print("\tTraining Accuracy: {}".format(sess.run(accuracy,
                                                            feed_dict_t)))
            print("\tValidation Cost: {}".format(sess.run(loss, feed_dict_v)))
            print("\tValidation Accuracy: {}".format(sess.run(accuracy,
                                                              feed_dict_v)))
            global_step += step - 1
            step = 1
            if epoch != 0:
                alpha = learning_rate_decay(alpha, decay_rate, epoch, 1)
            layer_size_activation_index = 0
            batch_start = 0
            batch_stop = batch_size
            if epoch == epochs:
                break
            xt, yt = shuffle_data(Data_train[0], Data_train[1])
            while batch_start <= m:
                # batch creation
                current_batch_x = xt[batch_start: batch_stop]
                current_batch_y = yt[batch_start: batch_stop]
                feed_dict_b = {x: current_batch_x, y: current_batch_y}
                # batch evaluation
                if step % 100 == 0:
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(sess.run(loss, feed_dict_b)))
                    print("\t\tAccuracy: {}".format(sess.run(accuracy,
                                                             feed_dict_b)))
                # forward prop and train op
                sess.run(y_pred, feed_dict_b)
                sess.run(train_op, feed_dict_b)
                # end train op, increments follow
                layer_size_activation_index += 1
                step += 1
                batch_start = batch_stop
                batch_stop += batch_size
            epoch += 1
        save = saver.save(sess, save_path)
        return save_path
