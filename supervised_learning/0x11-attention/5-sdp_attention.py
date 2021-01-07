#!/usr/bin/env python3
""""""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """"""
    print(Q.shape, tf.size(Q))
    print(K.shape, tf.size(K))
    print(V.shape, tf.size(V))

    """
    my_shape_k = [K.shape[i] for i in range(len(K.shape) - 1)]
    my_shape_q = [Q.shape[i] for i in range(len(Q.shape) - 2)]

    my_shape_k.insert(0, K.shape[-1])
    my_shape_q.insert(0, Q.shape[-2])
    my_shape_q.append(Q.shape[-1])

    print(my_shape_q, my_shape_k)

    K = tf.reshape(K, (my_shape_k))
    Q = tf.reshape(Q, (my_shape_q))

    print(Q.shape, K.shape)
    """

    attn = tf.matmul(Q, K, transpose_b=True)

    if mask is not None:
        attn = tf.add(tf.multiply(mask(attn), -1e9), attn)
    
    attn = tf.keras.layers.Softmax(axis=-1)(attn)

    output = tf.matmul(attn, V)

    return output, attn
