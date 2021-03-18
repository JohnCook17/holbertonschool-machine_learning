#!/usr/bin/env python3
"""Finds topics in the collection"""


def schools_by_topic(mongo_collection, topic):
    """finds the desiered topic"""
    return mongo_collection.find({"topics": topic})
