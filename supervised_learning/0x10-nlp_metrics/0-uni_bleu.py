#!/usr/bin/env python3
""""""
import numpy as np


def count_ngram(unigram, ngram=1):
    """counts and groups ngrams"""
    my_count = {}
    for i in range(len(unigram)):
        my_ngram = unigram[i : i + ngram]
        # print(my_ngram)
        new_ngram = ""
        for j, word in enumerate(my_ngram):
            new_ngram += str(word)
            if j != len(my_ngram) and ngram > 1:
                new_ngram += " "
        # print(new_ngram)
        if new_ngram not in my_count.keys():
            my_count[new_ngram] = 1
        else:
            my_count[new_ngram] += 1

    print(my_count)

    return my_count


def modified_precision(references, sentence, n):
    """"""
    counts = count_ngram(sentence, n)
    max_counts = {}

    for ref in references:
        ref_ngram = count_ngram(ref, n)
        for ngram in ref_ngram.keys():
            max_counts[ngram] = max(max_counts.get(ngram, 0), ref_ngram[ngram])
    
    clipped_counts = {ngram: min(count, max_counts.get(ngram, 0)) for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())

    denominator = max(1, sum(counts.values()))

    print(numerator / denominator)

    return numerator, denominator


def uni_bleu(references, sentence):
    """"""
    n = 1
    p_num = {}
    p_den = {}
    sen_lens = 0
    ref_lengths = 0
    weights = [1 / n for i in range(n)] # change this to 1 / n n times
    print(weights)
    print("===========")
    for ref in references:
        for i in range(1, len(weights) + 1):
            p_num[i], p_den[i] = modified_precision(references, sentence, n)
            sen_len = len(sentence)
            sen_lens += sen_len
            ref_lengths += (np.abs(np.asarray([len(x) for x in references]) - len(sentence))).argmin()

    best = (np.abs(np.asarray([len(x) for x in references]) - len(sentence))).argmin()

    p_n = [p_num[i] / p_den[i] for i in range(1, len(weights) + 1)]

    # brevity penalty
    if len(sentence) > best:
        bp = 1
    elif len(sentence) <= best:
        bp = np.exp(1 - (ref_lengths / sen_lens))
    # end brevity penalty

    s = (w_i * np.log(p_i) for w_i, p_i in zip(weights, p_n))
    print("bp = ", bp)
    s = np.exp(np.sum(s))
    print("s = ", s)
    bleu = bp * s

    return bleu
