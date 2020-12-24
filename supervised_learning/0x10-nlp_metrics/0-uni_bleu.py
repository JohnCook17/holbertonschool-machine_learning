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


def modified_precision(ref, sentence, n):
    """"""
    counts = count_ngram(sentence, n)
    max_counts = {}

    # for ref in references:
    ref_ngram = count_ngram(ref, n)
    for ngram in counts:
        max_counts[ngram] = max(max_counts.get(ngram, 0), ref_ngram.get(ngram, 0))
    
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())

    denominator = max(1, sum(counts.values()))

    print(numerator / denominator)

    return numerator, denominator


def uni_bleu(references, sentence):
    """"""
    n = 1
    weights = [1 / n for i in range(n)] # change this to 1 / n n times
    p_num = 0
    p_den = 0
    p_num_total = {i: 0 for i in range(1, len(weights) + 1)}
    p_den_total = {i: 0 for i in range(1, len(weights) + 1)}
    sen_lens = 0
    ref_lengths = 0
    print(weights)
    print("===========")
    for ref in references:
        for i in range(1, len(weights) + 1):
            p_num, p_den = modified_precision(ref, sentence, n)
            print(p_num, p_den)
            p_num_total[i] += p_num
            p_den_total[i] += p_den
            sen_len = len(sentence)
            sen_lens += sen_len
            ref_lengths += (np.abs(np.asarray([len(x) for x in references]) - len(sentence))).argmin()

    best = (np.abs(np.asarray([len(x) for x in references]) - len(sentence))).argmin()

    print(p_num_total, p_den_total)

    p_n = [p_num_total[i] / p_den_total[i] for i in range(1, len(weights) + 1)]

    # brevity penalty
    if len(sentence) > best:
        bp = 1
    elif len(sentence) <= best:
        bp = np.exp(1 - (ref_lengths / sen_lens))
    # end brevity penalty

    print("p_n = ", p_n)
    s = (w_i * np.log(p_i) for w_i, p_i in zip(weights, p_n))
    print("bp = ", bp)
    s = np.exp(np.sum(s))
    print("s = ", s)
    bleu = bp * s

    return bleu
