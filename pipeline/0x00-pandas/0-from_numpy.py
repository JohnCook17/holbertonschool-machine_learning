#!/usr/bin/env python3
"""takes a numpy array and makes a dataframe"""
import pandas as pd


def from_numpy(array):
    """makes a df"""
    return pd.DataFrame(array)
