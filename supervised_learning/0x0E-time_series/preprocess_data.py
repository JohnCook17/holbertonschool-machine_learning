#!/usr/bin/env python3

import numpy as np
import pandas as pd
from os import path

def preprocess(df_to_load):
    """"""

    print("====================LOADING DATA====================")
    df = pd.read_csv(df_to_load, parse_dates=["Timestamp"])

    print(df.dtypes)

    print("====================PROCESSING DATA====================")

    print("Dropping NAN's")
    df = df.dropna()

    print("Converting to year, month day, time format")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")

    print("======================")
    print(df.dtypes)
    print("======================")

    print("Removing unneeded values")
    df = df.drop(labels=["Open", "High", "Low", "Volume_(Currency)", "Weighted_Price"], axis=1)

    print("======================")
    print(df.dtypes)
    print("======================")

    print("======================")
    print(df.head())
    print("======================")

    print("Getting last day")
    coin_base_date = pd.to_datetime(df["Timestamp"]).values[-1]

    start_time = coin_base_date - np.timedelta64(1, "D")
    end_time = coin_base_date
    df = df[df.Timestamp.between(start_time, end_time)]
    
    print("======================")
    print(df.head())
    print("======================")

if __name__ == "__main__":

    dfs = ["data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"]
    for data in dfs:
        preprocess(data)
