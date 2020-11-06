#!/usr/bin/env python3
"""
Write a function def kmeans(X, k, iterations=1000): that performs K-means on a dataset:

X is a numpy.ndarray of shape (n, d) containing the dataset
n is the number of data points
d is the number of dimensions for each data point
k is a positive integer containing the number of clusters
iterations is a positive integer containing the maximum number of iterations that should be performed
If no change in the cluster centroids occurs between iterations, your function should return
Initialize the cluster centroids using a multivariate uniform distribution (based on0-initialize.py)
If a cluster contains no data points during the update step, reinitialize its centroid
You should use numpy.random.uniform exactly twice
You may use at most 2 loops
Returns: C, clss, or None, None on failure
C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
clss is a numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to

expected output =
[[ 9.92511389 25.73098987]
 [30.06722465 40.41123947]
 [39.62770705 19.89843487]
 [59.22766628 29.19796006]
 [20.0835633  69.81592298]]
"""
import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
