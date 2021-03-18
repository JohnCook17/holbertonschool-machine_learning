#!/usr/bin/env python3
"""updates all that match name with topics"""


def update_topics(mongo_collection, name, topics):
    """updateds the topics of the collection"""
    return mongo_collection.update_many({"name": name}, {"$set": {"topics":topics}})