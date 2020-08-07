#!/usr/bin/env python3
"""performes agglomerative clustering"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """performes agglomerative clustering and makes a dendrogram"""
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')
    scipy.cluster.hierarchy.dendrogram(Z,
                                       leaf_rotation=90.,
                                       color_threshold=dist)
    plt.show()
    clusters = scipy.cluster.hierarchy.fcluster(Z, dist, criterion="distance")
    return clusters
