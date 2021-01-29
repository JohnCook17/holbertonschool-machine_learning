#!/usr/bin/env python3
"""Searches multiple articles for an answer"""
import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """Searches multiple articles for an answer
       corpus_path is the path to the docs
       sentence is the question or sentence trying to find best match for
    """
    m = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    articles = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            articles.append(f.read())

    embeddings = m(articles)

    corr = np.inner(embeddings, embeddings)

    closest = np.argmax(corr[0, 1:])

    return articles[closest + 1]
