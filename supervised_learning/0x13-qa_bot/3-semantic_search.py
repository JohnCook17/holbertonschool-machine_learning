#!/usr/bin/env python3
""""""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel


tokenizer = (BertTokenizer
             .from_pretrained('bert-large-uncased-whole-word-masking'))
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def make_tokens(reference):
    """"""
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = paragraph_tokens

    return tokens, paragraph_tokens


def bert(tokens, sentence_tokens, paragraph_tokens):
    """"""
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_masks = [1] * len(input_word_ids)
    input_type_ids = ([0] * (1 + len(sentence_tokens) + 1) + [1]
                      * (len(paragraph_tokens) + 1))

    print(len(input_word_ids), len(input_masks), len(input_type_ids))

    start = 0
    max_size = 511
    total_size = len(input_word_ids)
    answers = []

    while total_size > 511:

        (input_word_id,
            input_mask,
            input_type_id) = map(lambda t:
                                 tf.expand_dims(tf.convert_to_tensor(t[start:max_size], dtype=tf.int32), axis=0),  # make into loop of size 512 each
                                 (input_word_ids,
                                  input_masks,
                                  input_type_ids))
        outputs = model([input_word_id, input_mask, input_type_id])
        # using `[1:]` will enforce an answer. `outputs[0][0][0]`
        # is the ignored '[CLS]' token logit
        short_start = tf.argmax(outputs[0][0][len(sentence_tokens):]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        start += 511
        max_size += 511
        total_size -= 511

        if answer:
            print("There is an answer!!!")
            print(answer)
            answers.append(answer)

    return answers


def semantic_search(corpus_path, sentence):
    """"""
    files = tf.io.gfile.walk(corpus_path)

    try:
        tf.io.gfile.remove(corpus_path + "/" + ".DS_Store")
    except Exception as e:
        print("\nNo DS_Store found\n")

    sentence_tokens = tokenizer.tokenize(sentence)

    # print(len(sentence_tokens))

    sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]
    paragraph_tokens = []  # ["[CLS]"] + ["[SEP]"]
    tokens = ["[SEP]"] + sentence_tokens + ["[SEP]"]

    for my_file in files:
        for md in my_file[2]:
            with open(corpus_path + "/" + str(md)) as f:
                reference = f.read()
                token, paragraph_token = make_tokens(reference)
                # print(len(token))
                # print(len(paragraph_token))
                tokens += token + ["[SEP]"]
                # sentence_tokens += sentence_token  # might need to remove!
                paragraph_tokens += paragraph_token + ["[SEP]"]

    tokens += ["[SEP]"]
    # paragraph_tokens += ["[SEP]"]

    print(tokens, len(sentence_tokens), len(paragraph_tokens))

    answer = bert(tokens, sentence_tokens, paragraph_tokens)

    return answer
