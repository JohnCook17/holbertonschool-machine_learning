#!/usr/bin/env python3
"""List all starwars planets with sentient life"""
import requests


def sentientPlanets():
    """list planets with sentient life"""
    r = requests.get("https://swapi-api.hbtn.io/api/species").json()

    next_page = r["next"]

    planets = {}

    while next_page != "null":
        for species in r["results"]:
            if ((species["homeworld"] != "n/a" and
                 species["homeworld"] is not None)):
                planet = requests.get(species["homeworld"]).json()
                planets[planet["name"]] = species
        # print(next_page)
        if next_page == "null" or next_page is None:
            break
        r = requests.get(next_page).json()

        next_page = r["next"]

    return planets.keys()
