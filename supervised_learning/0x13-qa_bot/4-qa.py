#!/usr/bin/env python3
"""Asnwers multiple questions from multiple documents"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """Searches multiple articles for an answer
       corpus_path is the path to the docs
       sentence is the question or sentence trying to find best match for
    """
    tokenizer = (BertTokenizer
                 .from_pretrained('bert-large-uncased-whole-word-masking'))
    m = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    articles = [sentence]

    for filename in os.listdir(corpus_path):
        if not filename.endswith(".md"):
            continue
        with open(corpus_path + "/" + filename, "r", encoding="utf-8") as f:
            articles.append(f.read())

    embeddings = m(articles)

    corr = np.inner(embeddings, embeddings)

    print(corr)

    closest = np.argmax(corr[0, 1:])

    return articles[closest + 1]


def text_search(question, reference):
    """Answers a question
       question is the question to answer
       reference is the doc to read from
    """
    tokenizer = (BertTokenizer
                 .from_pretrained('bert-large-uncased-whole-word-masking'))
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = (["[CLS]"] + question_tokens + ["[SEP]"]
              + paragraph_tokens + ["[SEP]"])
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = ([0] * (1 + len(question_tokens) + 1) + [1]
                      * (len(paragraph_tokens) + 1))

    input_word_ids, input_mask, input_type_ids = map(lambda t:
                                                     tf.expand_dims
                                                     (tf.convert_to_tensor
                                                      (t, dtype=tf.int32), 0),
                                                     (input_word_ids,
                                                      input_mask,
                                                      input_type_ids))
    outputs = model([input_word_ids, input_mask, input_type_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]`
    # is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer


def answer_question(question, reference):
    """A wrapper function for semantic search and txt to search
       returns the answer to the question"""
    txt_to_search = semantic_search(reference, question)
    return text_search(question, txt_to_search)


def qa_bot(reference):
    """input output loop
       reference is the path to the articles to search for an answer
    """
    while True:
        print("Q: ", end="")
        question = input().lower()

        if (("exit" in question.lower().strip()
             or "quit" in question.lower().strip()
             or "bye" in question.lower().strip())):
            print("A: Goodbye")
            exit()

        answer = answer_question(question, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: ", answer)
