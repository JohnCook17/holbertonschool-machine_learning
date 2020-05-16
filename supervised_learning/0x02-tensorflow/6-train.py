#!/usr/bin/env python3
"""Runs the training of the neural network"""
import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    X_train: the input data in a np.ndarray
    Y_train: the training lables in a np.ndarray
    X_valid: the validation data
    Y_valid: the validation lables
    layer_sizes: a list of the size of each layer
    activations: the activation to use per layer
    alpha: the learning rate
    iterations: the number of times to train
    save_path: save file location
    """

    def create_placeholders(nx, classes):
        """x is the input data placeholder,
        y is for the onehot lables for input data"""
        x = tf.placeholder(float, [None, nx])
        y = tf.placeholder(float, [None, classes])
        return x, y

    def create_layer(prev, n, activation):
        """prev is the previous layer, n is the number of nodes,
        activation is the activation, i is for initialzation, which
        uses he et al"""
        i = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        layer = tf.layers.Dense(units=n, activation=activation,
                                kernel_initializer=i)
        return layer(prev)

    def forward_prop(x, layer_sizes=[], activations=[]):
        """x is the placeholder for input, layer_sizes is a list of layer
        sizes, and activations is a list of activation functions to use"""
        prev = x
        for layer, activ in zip(layer_sizes, activations):
            prev = create_layer(prev, layer, activ)
        return prev

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

    def create_train_op(loss, alpha):
        """Trains the neural network using gradient descent minimizing loss"""
        return tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    with tf.Session() as sess:
        iteration = 0
        x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
        y_pred = forward_prop(x, layer_sizes, activations)
        loss = calculate_loss(y, y_pred)
        accuracy = calculate_accuracy(y, y_pred)
        train_op = create_train_op(loss, alpha)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        tf.add_to_collection("x_placeholder", x)
        tf.add_to_collection("y_placeholder", y)
        tf.add_to_collection("y_pred", y_pred)
        tf.add_to_collection("accuracy", accuracy)
        tf.add_to_collection("loss", loss)
        while iteration <= iterations:
            if iteration == 0 or iteration % 100 == 0 or (iteration ==
                                                          iterations):
                print("After {} iterations:".format(iteration))
                print("\tTraining Cost: {}"
                      .format(sess.run(loss,
                                       feed_dict={x: X_train, y: Y_train})))
                print("\tTraining Accuracy: {}"
                      .format(sess.run(accuracy,
                                       feed_dict={x: X_train, y: Y_train})))
                print("\tValidation Cost: {}"
                      .format(sess.run(loss,
                                       feed_dict={x: X_valid, y: Y_valid})))
                print("\tValidation Accuracy: {}"
                      .format(sess.run(accuracy,
                                       feed_dict={x: X_valid, y: Y_valid})))
            sess.run(y_pred, feed_dict={x: X_train, y: Y_train})
            sess.run(y_pred, feed_dict={x: X_valid, y: Y_valid})
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            iteration += 1
        save = saver.save(sess, save_path)
        return save
