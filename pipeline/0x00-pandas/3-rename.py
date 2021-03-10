#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# print(df.tail())

df = df.drop(labels=["Open",
                     "High",
                     "Low",
                     "Volume_(BTC)",
                     "Volume_(Currency)",
                     "Weighted_Price"],
             axis=1)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
df = df.rename(columns={"Timestamp": "Datetime", "Close": "Close"})

print(df.tail())
