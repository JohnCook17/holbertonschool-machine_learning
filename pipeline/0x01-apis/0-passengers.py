#!/usr/bin/env python3
""""""
import requests


def availableShips(passengerCount):
    """"""
    r = requests.get("https://swapi-api.hbtn.io/api/starships").json()

    next_page = r["next"]

    ships = []

    while next_page != "null":
        for ship in r["results"]:
            if ship["passengers"] != "n/a" and ship["passengers"] != "unknown":
                if int(ship["passengers"].replace(",", "")) >= passengerCount:
                    ships.append(ship["name"])
        # print(next_page)
        if next_page == "null" or next_page is None:
            break
        r = requests.get(next_page).json()

        next_page = r["next"]

    return ships
