#!/usr/bin/env python3
"""The Binomial Distribution"""


class Binomial():
    """The Binomial Class"""
    def __init__(self, data=None, n=1, p=0.5):
        """Init of n, for number of trials, and p, probability of success"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = ((1 / n) * sum([((data[i] - mean) ** 2)
                                       for i in range(0, len(data))]
                                      )) / len(data)
            self.p = -(variance / mean) + 1
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def factorial(self, n):
        """n factorial"""
        if n % 1 != 0:
            n = int(n)
        fact = 1
        for i in range(1, n + 1):
            fact = fact * i
        return fact

    def pmf(self, k):
        """The PMF of a binomial distribution"""
        if k % 1 != 0:
            k = int(k)
        if k < 0 and k <= self.n:
            return 0
        q = 1 - self.p
        co = (self.factorial(self.n) / ((self.factorial(self.n-k)
                                         * self.factorial(k))))
        q2 = q ** (self.n - k)
        return co * (self.p ** k) * q2

    def cdf(self, k):
        """cdf for binomial"""
        if k % 1 != 0:
            k = int(k)
        if k < 0 and k <= self.n:
            return 0
        return sum([self.pmf(i) for i in range(0, k + 1)])
