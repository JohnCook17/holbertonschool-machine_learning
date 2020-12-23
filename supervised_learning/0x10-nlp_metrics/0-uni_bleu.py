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


def count_clip_ngram(references, sentence, n):
    """"""
    res = {}
    
    sent_ngram = count_ngram(sentence, n)
    for ref in references:
        ref_ngram = count_ngram(ref, n)
        for ngram in ref_ngram.keys():
            if ngram in res.keys():
                # print(ngram)
                res[ngram] = max(ref_ngram[ngram], res[ngram])
            else:
                # print(ngram)
                res[ngram] = ref_ngram[ngram]
    
    return {
        k: min(sent_ngram.get(k, 0), res.get(k, 0))
        for k in sent_ngram
    }


def uni_bleu(references, sentence):
    """"""
    n = 1
    new_dict = {}
    ct = 0
    
    print("===========")
    for ref in references:
        my_dict = count_ngram(ref, n)
        new_dict.update(my_dict)
    values = new_dict.values()
    ct = sum(values)
    denominator = float(max(ct, 1))
    my_p = []
    for word in sentence:
        ct_clip = count_clip_ngram(references, [word], n)
        numerator = sum(ct_clip.values())

        if numerator != 0:
            my_p.append(numerator / denominator)
        else:
            my_p.append(1e-10)

        denominator -= 1

    p_sum = 0
    for n in range(1, 5):
        p_sum += 1 / n * np.log(my_p[n - 1])

    p_sum = np.exp(p_sum)

    best = (np.abs(np.asarray([len(x) for x in references]) - len(sentence))).argmin()

    if len(sentence) > best:
        bp = 1
    elif len(sentence) <= best:
        bp = np.exp(1 - (len(sentence) / best))

    print(my_p)

    bleu = bp * p_sum

    return bleu
