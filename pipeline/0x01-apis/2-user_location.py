#!/usr/bin/env python3
"""uses github api to find the location if rate limit is not reached"""
import requests
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("a")
args = parser.parse_args()

r = requests.get(args.a)

if r.status_code == 200:
    print(r.json()["location"])
elif r.status_code == 403:
    minutes = pd.Timedelta(pd.to_datetime
                           (int(r.headers
                                .get("X-RateLimit-Reset")), unit="s")
                           - pd.Timestamp.now()).total_seconds() / 60
    print("Reset in {} min".format(int(minutes)))
else:
    print("Not found")
