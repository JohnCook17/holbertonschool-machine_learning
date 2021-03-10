#!/usr/bin/env python3
"""Fills Nans with 0 or previous close"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop("Weighted_Price", axis=1)

# remove the below line for actual assignment
# df = df.head(10000)

if df["Close"].isna()[0]:
    df.loc[0, "Close"] = 0

if df["High"].isna()[0]:
    df.loc[0, "High"] = df.loc[0, "Close"]
if df["Low"].isna()[0]:
    df.loc[0, "Low"] = df.loc[0, "Close"]
if df["Open"].isna()[0]:
    df.loc[0, "Open"] = df.loc[0, "Close"]

if df["Volume_(BTC)"].isna()[0]:
    df.loc[0, "Volume_(BTC)"] = 0
if df["Volume_(Currency)"].isna()[0]:
    df.loc[0, "Volume_(Currency)"] = 0

for i in range(1, len(df)):
    # print(i)
    if df["High"].isna()[i]:
        df.loc[i, "High"] = df.loc[i-1, "Close"]
    if df["Low"].isna()[i]:
        df.loc[i, "Low"] = df.loc[i-1, "Close"]
    if df["Open"].isna()[i]:
        df.loc[i, "Open"] = df.loc[i-1, "Close"]
    if df["Close"].isna()[i]:
        df.loc[i, "Close"] = df.loc[i-1, "Close"]
    if df["Volume_(BTC)"].isna()[i]:
        df.loc[i, "Volume_(BTC)"] = 0
    if df["Volume_(Currency)"].isna()[i]:
        df.loc[i, "Volume_(Currency)"] = 0


"""def fill_na(df):

    prev_value = df.loc[0, "Close"]

    def find_na(row):
        print(row)
        nonlocal prev_value
        new_value = prev_value
        if pd.isna(row["High"]):
            row["High"] = new_value
        if pd.isna(row["Low"]):
            row["Low"] = new_value
        if pd.isna(row["Open"]):
            row["Open"] = new_value
        if pd.isna(row["Close"]):
            row["Close"] = new_value

        if pd.isna(row["Volume_(BTC)"]):
            row["Volume_(BTC"] = 0.0
        if pd.isna(row["Volume_(Currency)"]):
            row["Currency"] = 0.0

        prev_value = row["Close"]
        return row

    df.iloc[1:] = df.iloc[1:].apply(find_na, axis=1)

    return df


df = fill_na(df)"""

print(df.head())
print(df.tail())
