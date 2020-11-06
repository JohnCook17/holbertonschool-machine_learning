#!/usr/bin/env python3
"""Agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """uses ward linkage"""
    Z = scipy.cluster.hierarchy.linkage(X, "ward")
    fig = plt.figure()
    dn = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion="distance")
    return clss
