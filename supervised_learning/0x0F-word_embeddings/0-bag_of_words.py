#!/usr/bin/env python3
"""Bag of words"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """sentences is a list of sentences to analyze
       vocab is a list of vocabulary words to use for analysis"""
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        sentences[i] = sentences[i].replace("'s", "")
        sentences[i] = re.sub(r'\W', ' ', sentences[i])
        sentences[i] = re.sub(r'\s+', ' ', sentences[i])

    if vocab is None:
        vocab = {}
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1

    words = {}
    for sentence in sentences:
        # print(sentence)
        for word in sentence.split():
            # print(word)
            if word not in vocab.keys():
                continue
            elif word in vocab.keys() and word not in words.keys():
                words[word] = 1
            else:
                words[word] += 1

    # print(words)

    features = sorted(words.keys())

    BoW = np.zeros((len(sentences), len(features)))
    for j, sentence in enumerate(sentences):
        for i, word in enumerate(features):
            if word in sentence:
                BoW[j, i] = sentence.split().count(word)

    BoW = BoW.astype("int32")

    return BoW, features
