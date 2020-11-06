#!/usr/bin/env python3
"""k-means in sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """The k-means function form sklearn"""
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return k_means.cluster_centers_, k_means.labels_
