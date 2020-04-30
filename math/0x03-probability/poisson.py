#!/usr/bin/env python3
"""The Poisson distribution"""


class Poisson:
    """The Poisson class"""
    def __init__(self, data=None, lambtha=1.):
        """Init lambtha of data"""
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
            mean = (sum(data) / len(data))
            self.lambtha = mean

    def factorial(self, n):
        """n factorial"""
        if n % 1 != 0:
            n = int(n)
        fact = 1
        for i in range(1, n + 1):
            fact = fact * i
        return fact

    def pmf(self, k):
        """The probability mass function of Poisson"""
        if not isinstance(k, int):
            k = int(k)
        e = 2.7182818285

        fact = 1
        for i in range(1, k + 1):
            fact = fact * i

        mean = self.lambtha
        return (((e ** (-mean)) * (mean ** k)) / self.factorial(k))

    def cdf(self, k):
        """The Cumulative distribution function of poisson"""
        if k % 1 != 0:
            k = int(k)
        e = 2.7182818285
        mean = self.lambtha
        k_sum = 0
        i = 0
        while i <= k:
            k_sum += mean ** i / self.factorial(i)
            i += 1
        return (e ** (-mean)) * k_sum
