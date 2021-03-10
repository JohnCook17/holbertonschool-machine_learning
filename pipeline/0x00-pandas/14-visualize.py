#!/usr/bin/env python3
"""Visualizes the data from start_date to end_date"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def fill(df):
    """fills data with pervious close or 0"""
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df["Timestamp"] = df["Timestamp"].dt.to_period("d")
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    df["Date"] = df["Timestamp"]

    # df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("Date")

    df = df.drop_duplicates(subset="Timestamp", keep="last")

    start_date = "2017-01-01"
    end_date = "2021-03-19"

    after_start_date = df["Timestamp"] >= start_date
    before_end_date = df["Timestamp"] <= end_date
    between_two_dates = after_start_date & before_end_date
    df = df.loc[between_two_dates]

    # print(df.head(), df.tail())

    df = df.drop(["Timestamp", "Weighted_Price"], axis=1)

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
        """fills na with previous close"""
        nonlocal prev_value
        new_value = prev_value
        if pd.isna(row[col]):
            ret = new_value
        else:
            ret = row[col]
        prev_value = row["Close"]
        return ret

    def fill_volume(row):
        """fills na with 0"""
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

    print(df)

    df.plot()
    plt.show()


fill(df)
