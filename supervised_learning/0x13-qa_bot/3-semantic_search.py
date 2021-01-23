#!/usr/bin/env python3
""""""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel

tokenizer = (BertTokenizer
             .from_pretrained('bert-large-uncased-whole-word-masking'))
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def bert(reference, sentence):
    """"""
    sentence_tokens = tokenizer.tokenize(sentence)
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = (["[CLS]"] + sentence_tokens + ["[SEP]"]
              + paragraph_tokens + ["[SEP]"])

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_masks = [1] * len(input_word_ids)
    input_type_ids = ([0] * (1 + len(sentence_tokens) + 1) + [1]
                      * (len(paragraph_tokens) + 1))

    # print(len(input_word_ids), len(input_masks), len(input_type_ids))

    (input_word_id,
        input_mask,
        input_type_id) = map(lambda t:
                             tf.expand_dims(tf.convert_to_tensor
                                            (t, dtype=tf.int32), axis=0),
                             (input_word_ids,
                              input_masks,
                              input_type_ids))
    outputs = model([input_word_id, input_mask, input_type_id])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]`
    # is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    # short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start:]  # short_end + 1]

    if answer_tokens:
        return answer_tokens


def semantic_search(corpus_path, sentence):
    """"""
    files = tf.io.gfile.walk(corpus_path)

    try:
        tf.io.gfile.remove(corpus_path + "/" + ".DS_Store")
    except Exception as e:
        print("\nNo DS_Store found\n")

    answers = []

    for my_file in files:
        files_len = len(my_file[2])
        for i, md in enumerate(my_file[2]):
            print("\nWorking on file {} out of {}!!!\n".format(i, files_len))
            with open(corpus_path + "/" + str(md)) as f:
                reference = f.read()
                answers.append(bert(reference, sentence))

    final_answer = tf.argmax(answers)
    final_answer = tokenizer.convert_tokens_to_string(final_answer)

    return final_answer
