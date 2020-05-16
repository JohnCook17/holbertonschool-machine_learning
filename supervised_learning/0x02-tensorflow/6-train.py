#!/usr/bin/env python3
"""Runs the training of the neural network"""
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


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
        tf.add_to_collection("x", x)
        tf.add_to_collection("y", y)
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
        return save_path
