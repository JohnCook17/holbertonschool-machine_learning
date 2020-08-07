#!/usr/bin/env python3
"""kmeans in sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """kmeans in sklearn"""
    KMeans = sklearn.cluster.KMeans(k).fit(X)
    C = KMeans.cluster_centers_
    clss = KMeans.labels_

    return C, clss
