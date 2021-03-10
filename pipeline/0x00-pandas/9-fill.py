#!/usr/bin/env python3
"""Fills Nan's """
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop("Weighted_Price", axis=1)


def fill(df):
    """Fills Nan's with pervious close or 0 depending on which is relevant"""
    if df["Close"].isna()[0]:
        df.loc[0, "Close"] = 0

    df["Close1"] = df["Close"]

    if df["High"].isna()[0]:
        df.loc[0, "High"] = df.loc[0, "Close"]
    if df["Low"].isna()[0]:
        df.loc[0, "Low"] = df.loc[0, "Close"]
    if df["Open"].isna()[0]:
        df.loc[0, "Open"] = df.loc[0, "Close"]

    prev_value = df["Close"][0]

    def fill_na(row, col):
        """Fills Nan's with pervious close"""
        nonlocal prev_value
        new_value = prev_value
        if pd.isna(row[col]):
            ret = new_value
        else:
            ret = row[col]
        prev_value = row["Close"]
        return ret

    def fill_volume(row):
        """Fills Nan's with 0"""
        if pd.isna(row):
            ret = 0
        else:
            ret = row
        return ret

    df.iloc[1:]["Open"] = df.iloc[1:][["Open", "Close"]].apply(fill_na,
                                                               axis=1,
                                                               args=["Open"])
    df.iloc[1:]["High"] = df.iloc[1:][["High", "Close"]].apply(fill_na,
                                                               axis=1,
                                                               args=["High"])
    df.iloc[1:]["Low"] = df.iloc[1:][["Low", "Close"]].apply(fill_na,
                                                             axis=1,
                                                             args=["Low"])
    df.iloc[1:]["Close1"] = df.iloc[1:][["Close1",
                                         "Close"]].apply(fill_na,
                                                         axis=1,
                                                         args=["Close1"])

    df.iloc[1:]["Volume_(BTC)"] = (df.iloc[1:]["Volume_(BTC)"]
                                   .apply(fill_volume))
    df.iloc[1:]["Volume_(Currency)"] = (df.iloc[1:]["Volume_(Currency)"]
                                        .apply(fill_volume))

    df["Close"] = df["Close1"]

    df = df.drop("Close1", axis=1)


print(df.head())
print(df.tail())
