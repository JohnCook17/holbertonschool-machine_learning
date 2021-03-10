#!/usr/bin/env python3
""""""
import pandas as pd


my_dict = {"First": [0.0, 0.5, 1.0, 1.5],
           "Second": ["one", "two", "three", "four"]}

df = pd.DataFrame(my_dict, index=["A", "B", "C", "D"])
