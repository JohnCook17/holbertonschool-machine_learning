#!/usr/bin/env python3
"""mini batch gradient descent"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a mini batch of gradient descent"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + ".meta")
        new_saver.restore(sess, load_path)
        m = X_train.shape[0]
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        epoch = 0
        while epoch <= epochs:
            xt, yt = shuffle_data(X_train, Y_train)
            xv, yv = X_valid, Y_valid
            feed_dict_t = {x: xt, y: yt}
            feed_dict_v = {x: xv, y: yv}
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(sess.run(loss, feed_dict_t)))
            print("\tTraining Accuracy: {}".format(sess.run(accuracy, feed_dict_t)))
            print("\tValidation Cost: {}".format(sess.run(loss, feed_dict_v)))
            print("\tValidation Accuracy: {}".format(sess.run(accuracy, feed_dict_v)))
            if epoch == epochs:
                break
            step = 0
            batch_start = 0
            batch_stop = batch_size
            while batch_start <= m:
                current_batch_x = xt[batch_start: batch_stop]
                current_batch_y = yt[batch_start: batch_stop]
                current_batch = {x: current_batch_x, y: current_batch_y}
                # print(len(current_batch_x))
                # print(len(current_batch_y))
                # print(batch_start)
                # print(batch_stop)
                if step % 100 == 0 and step >= 100:
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(sess.run(loss, current_batch)))
                    print("\t\tAccuracy: {}".format(sess.run(accuracy, current_batch)))
                sess.run(train_op, current_batch)
                step += 1
                batch_start = batch_stop
                batch_stop += batch_size
            epoch += 1
        tf.train.Saver().save(sess, save_path)
        return save_path
