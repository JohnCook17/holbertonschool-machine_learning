#!/usr/bin/env python3
"""Finds the next SpaceX launch"""
import requests


if __name__ == '__main__':
    r = requests.get("https://api.spacexdata.com/v4/launches/upcoming").json()
    launch_name = r[0]["name"]
    date_local = r[0]["date_local"]
    rocket_id = r[0]["rocket"]
    launchpad_id = r[0]["launchpad"]
    rocket_name = requests.get("https://api.spacexdata.com/v4/rockets/{}"
                               .format(rocket_id)).json()["name"]
    launchpad = requests.get("https://api.spacexdata.com/v4/launchpads/{}"
                             .format(launchpad_id)).json()["name"]

    print(launch_name, date_local, rocket_name, launchpad)
