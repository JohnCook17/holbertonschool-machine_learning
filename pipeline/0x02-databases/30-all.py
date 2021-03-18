#!/usr/bin/env python3
"""list all documents in mongo collection"""
import pymongo


def list_all(mongo_collection):
    """uses list comprehension to make a list of all docs"""
    return [doc for doc in mongo_collection.find({})]