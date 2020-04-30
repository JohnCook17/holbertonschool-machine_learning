#!/usr/bin/env python3
"""Normal Distribution"""


class Normal:
    """The Normal Class"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """init stddev and mean"""
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            pi = 3.1415926536
            self.mean = (sum(data)) / (len(data))
            x = self.mean
            n = len(data)
            new_data = [((data[i] - x) ** 2) for i in range(0, n)]
            self.stddev = ((1 / n) * sum(new_data)) ** .5

    def z_score(self, x):
        """Z-score of Normal deviation"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """x-value of z"""
        return ((self.stddev * z) + self.mean)

    def pdf(self, x):
        """pdf of normal"""
        e = 2.7182818285
        e_calc = e ** ((-1 / 2) * ((x - self.mean) / self.stddev) ** 2)
        pi = 3.1415926536
        pi_calc = (1 / (self.stddev * ((2 * pi) ** .5)))
        return pi_calc * e_calc

    def cdf(self, x):
        """cdf of normal"""
        pi = 3.1415926536
        x = ((x - self.mean) / (self.stddev * (2 ** .5)))
        erf = (2 / (pi ** .5)) * (x - ((x ** 3) / 3) + ((x ** 5) / 10) - ((x ** 7) / 42) + ((x ** 9) / 216))
        return .5 * (1 + erf)
