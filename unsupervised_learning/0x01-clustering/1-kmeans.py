#!/usr/bin/env python3
"""k-means clustering"""
import numpy as np


def initialize(X, k):
    """Initializes for K means given the data X"""
    try:
        if k < 1:
            return None
        d = X.shape[1]
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        return np.random.uniform(low=data_min, high=data_max, size=(k, d))
    except Exception as e:
        return None


def assign_class(X, centroids):
    """assigns the class"""
    return np.argmin(((X[:, :, None] - centroids.T[None, :, :, ]) ** 2)
                     .sum(axis=1), axis=1)


def get_centroids(X, k, clss):
    """gets the centroids"""
    centroids = []
    for j in range(k):
        if np.isclose(X[clss == j, :].any(), 0):
            centroids.append(np.random.uniform(np.min(X, axis=0),
                                               np.max(X, axis=0),
                                               (1, X.shape[1])))
        else:
            centroids.append(np.mean(X[clss == j, :], axis=0))
    """
    centroids = [np.random.uniform(np.min(X, axis=0),
                                   np.max(X, axis=0),
                                   (1, X.shape[1]))
                 if np.isclose(np.size(X[clss == j, :]), 0)
                 else np.mean(X[clss == j, :], axis=0)
                 for j in range(k)]
    """
    centroids = np.vstack(centroids)
    return centroids


def kmeans(X, k, iterations=1000):
    """k-means clustering"""
    try:
        if not isinstance(X, np.ndarray):
            return None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None
        n, d = X.shape
        if not isinstance(k, int) or k < 1 or k > n:
            return None, None

        centroids = initialize(X, k)
        for _ in range(iterations):
            clss = assign_class(X, centroids)
            new_centroids = get_centroids(X, k, clss)

            if np.array_equal(new_centroids, centroids):
                break
            else:
                centroids = np.copy(new_centroids)
        clss = assign_class(X, centroids)

        return new_centroids, clss

    except Exception as e:
        return None, None
