#!/usr/bin/env python3
""""""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """"""
    def __init__(self,
                 f,
                 X_init,
                 Y_init,
                 bounds,
                 ac_samples,
                 l=1,
                 sigma_f=1,
                 xsi=0.01,
                 minimize=True):
        """"""
        self.f = f
        self.gp = GP
        self.X_s = 