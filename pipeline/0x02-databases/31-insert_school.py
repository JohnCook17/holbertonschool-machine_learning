#!/usr/bin/env python3
"""inserts new info"""


def insert_school(mongo_collection, **kwargs):
    """inserts new information into the db collection school"""
    return mongo_collection.insert(kwargs)