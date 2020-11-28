#!/usr/bin/env python3
"""k-means clustering"""
import numpy as np


def initialize(X, k):
    """Initializes for K means given the data X"""
    if k < 1:
        return None
    try:
        d = X.shape[1]
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        return np.random.uniform(low=data_min, high=data_max, size=(k, d))
    except Exception as e:
        return None


def kmeans(X, k, iterations=1000):
    """k-means clustering"""
    centroids = X[k:]

    for i in range(iterations):

        clss = np.argmin(((X[:, :, None] - centroids.T[None, :, :, ])
                          ** 2).sum(axis=1), axis=1)

        centroids = initialize(X, k)
        new_centroids = np.array([X[j, :] ==
                                  np.random.uniform(np.min(X, axis=0),
                                  np.max(X, axis=0)(1, X.shape[1]))
                                  if X[clss == i].size == 0
                                  else X[clss == j, :].mean(axis=0)
                                  for j in range(k)])

        if new_centroids.all() == centroids.all():
            break
        else:
            centroids = new_centroids
    return centroids, clss
