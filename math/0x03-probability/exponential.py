#!/usr/bin/env python3
"""Exponential Distribution"""


class Exponential:
    """The Exponential Distribution class"""
    def __init__(self, data=None, lambtha=1.):
        """init lambtha of data"""
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = (1 / sum(data)) * len(data)

    def pdf(self, x):
        """The pdf for exponential distribution"""
        if x < 0:
            return 0
        else:
            e = 2.7182818285
            return (self.lambtha * (e ** -(self.lambtha ** x)))

    def cdf(self, x):
        """The cumulative distribution function"""
        if x < 0:
            return 0
        else:
            e = 2.7182818285
            return (1 - (e ** (-self.lambtha * x)))
