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
    df = df.drop(labels=["Open", "High", "Low", "Volume_(BTC)", "Weighted_Price"], axis=1)

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

    print("making prediction dataset")
    pred_date = pd.to_datetime(df["Timestamp"]).values[-1]

    p_start_time = pred_date - np.timedelta64(2, "D")
    p_end_time = pred_date - np.timedelta64(1, "D")

    p_df = df[df.Timestamp.between(start_time, end_time)].copy()
    print("++++++++++++++++++++++++++++")
    print(p_df.head(), "\n\n", p_df.tail())
    print("++++++++++++++++++++++++++++")

    # print("Normalizing data")
    # df["Close"] = df["Close"] * df["Volume_(Currency)"] / df["Volume_(Currency)"].sum()

    print("getting Hours in day")
    my_data = []
    for i in range(1, 26):
        start_time = coin_base_date - np.timedelta64(i, "h")
        end_time = coin_base_date - np.timedelta64(i - 1, "h")
        my_data.append(df[df.Timestamp.between(start_time, end_time)]["Close"].values[-1])

    my_p_data = []
    for i in range(1, 25):
        p_start_time = coin_base_date - np.timedelta64(i, "h")
        p_end_time = coin_base_date - np.timedelta64(i - 1, "h")
        my_p_data.append(p_df[p_df.Timestamp.between(p_start_time, p_end_time)]["Close"].values[-1])

    print("making new dfs")
    new_df = pd.DataFrame(my_data, columns=["inputs"])

    new_p_df = pd.DataFrame(my_p_data, columns=["inputs"])

    print("offsetting Hours by 1 for target data")
    new_df.drop_duplicates(subset="inputs", keep="last")
    new_p_df.drop_duplicates(subset="inputs", keep="last")
    targets_df = pd.DataFrame()
    targets_df["targets"] = new_df["inputs"].tail(1)
    # new_df["targets"] = ""
    # new_df.ix[0, "targets"] = new_df["inputs"].values[-1]
    # new_df.drop_duplicates(subset="Target", keep="first")
    new_df.drop(new_df.tail(1).index, inplace=True)
    # new_df["Target"] = new_df["Hour"].shift(periods=-1,fill_value=new_df["Hour"].values[-1])

    print("======================")
    print(new_df.head(25))
    print(targets_df.head(2))
    print(new_p_df.head(24))
    print("======================")

    new_df.to_pickle(df_to_load[:-4] + "_pickle")
    new_df.to_csv(df_to_load[:-4] + "_preprocessed.csv", index=False)
    targets_df.to_csv(df_to_load[:-4] + "targets_preprocessed.csv", index=False)
    new_p_df.to_csv(df_to_load[:-4] + "_prediction.csv", index=False)

if __name__ == "__main__":

    dfs = ["data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", "data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"]
    for data in dfs:
        preprocess(data)