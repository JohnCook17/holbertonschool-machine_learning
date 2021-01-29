#!/usr/bin/env python3
"""Answers mulptiple questions"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel


def question_answer(question, reference):
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


def answer_loop(reference):
    """A simple answer loop, with question answering
       reference is the reference to answer the question from
    """
    while True:
        print("Q: ", end="")
        question = input()

        if (("exit" in question.lower().strip()
             or "quit" in question.lower().strip()
             or "bye" in question.lower().strip())):
            print("A: Goodbye")
            exit()

        answer = question_answer(question, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: ", answer)
