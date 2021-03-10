#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# print(df.head())
df = df.drop(["Timestamp",
              "Open",
              "Close",
              "Volume_(Currency)",
              "Weighted_Price"],
             axis=1)

df = df.iloc[::60, :]

print(df.tail())
