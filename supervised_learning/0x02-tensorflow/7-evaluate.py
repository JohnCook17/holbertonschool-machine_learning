#!/usr/bin/env python3
"""Evaluates the neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_path + ".meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("./"))
        # op = sess.graph.get_operations()
        # [print(m.values()) for m in op][1]
        x = tf.get_collection("x_placeholder")[0]
        y = tf.get_collection("y_placeholder")[0]
        feed_dict = {x: X, y: Y}
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        return (sess.run(y_pred, feed_dict),
                sess.run(accuracy, feed_dict),
                sess.run(loss, feed_dict))
