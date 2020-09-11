#!/usr/bin/env python3
"""numpy preprocessing of data"""
import numpy as np


def forward_fill_col(arr):
    out = arr.copy()
    for col_idx in range(out.shape[1]):
        for row_idx in range(out.shape[0]):
            if np.isnan(out[row_idx, col_idx]):
                out[row_idx, col_idx] = out[row_idx - 1, col_idx]
    return out


def clean_data():
    """"""
    my_file = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    my_data = np.genfromtxt(my_file, delimiter=",")[1:]
    print(my_data.shape)
    # clean_data = np.nan_to_num(my_data, copy=True, nan=0.0)
    my_clean_data = forward_fill_col(my_data)
    print(my_clean_data.shape)
    np.savetxt(fname="clean_data.csv", X=my_clean_data, delimiter=",")
    return my_clean_data


def make_days(my_data):
    """"""
    i = 1
    end_data = 0
    while my_data.shape[0] >= 86400:
        print(my_data.shape)
        start_of_day, end_of_day = get_day(my_data, i)
        end_data += 86400
        i += 1
        print(end_of_day)
        my_data = my_data[:-(end_data - 1)]


def get_day(my_data, i):
    """gets 1 day of data"""
    try:
        # print(my_data)
        end_of_day = my_data[-1, 0]
        # print(end_of_day)
        start_of_day = end_of_day - 86400  # 1 day in unix timestamp
        day = my_data[my_data[:, 0] >= start_of_day]
        # print(day[:, 0])
        # day = day[day[:, 5] > 0]  # makes 0s go away
        day = np.delete(day, [1, 2, 3, 5, 6, 7], axis=1)  # pca
        # day[:, 0] = np.round((day[:, 0] / 18374), 10)
        # print(day[:, 0])
        # print(day)
        # print(day.shape)
        np.savetxt(fname="clean_data_t-{}.csv".format(i), X=day, delimiter=",")
        return start_of_day, end_of_day
    except Exception as e:
        print(e)
        print("end of array, exiting...")
        exit()


if __name__ == "__main__":
    my_data = clean_data()
    make_days(my_data)
