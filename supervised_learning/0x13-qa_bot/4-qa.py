#!/usr/bin/env python3
""""""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel


def semantic_search(corpus_path, sentence):
    """"""
    tokenizer = (BertTokenizer
                 .from_pretrained('bert-large-uncased-whole-word-masking'))
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    files = tf.io.gfile.walk(corpus_path)

    VERBOSE = True

    try:
        tf.io.gfile.remove(corpus_path + "/" + ".DS_Store")
    except Exception as e:
        print("\nNo DS_Store found\n")

    outputs = []
    max_max = 0
    best_candidate = None

    for my_file in files:
        files_len = len(my_file[2])
        for i, md in enumerate(my_file[2]):
            with open(corpus_path + "/" + str(md)) as f:
                reference = f.read()

                my_table = str.maketrans("", "", ".!?")
                reference = reference.translate(my_table)

                sentence_tokens = tokenizer.tokenize(sentence)
                paragraph_tokens = tokenizer.tokenize(reference)
                tokens = (sentence_tokens + ["[SEP]"]
                          + paragraph_tokens)

                input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_masks = [1] * len(input_word_ids)
                input_type_ids = ([0] * (len(sentence_tokens) + 1) + [1]
                                  * (len(paragraph_tokens)))

                (input_word_id,
                    input_mask,
                    input_type_id) = map(lambda t:
                                         tf.expand_dims(tf.convert_to_tensor
                                                        (t, dtype=tf.int32),
                                                        axis=0),
                                                       (input_word_ids,
                                                        input_masks,
                                                        input_type_ids))
                output = model([input_word_id, input_mask, input_type_id])
                current_output = output[0][0][:]

                current_output = tf.clip_by_value(current_output,
                                                  clip_value_min=0,
                                                  clip_value_max=500)

                zero = tf.constant(0, dtype=tf.float32)
                one = tf.constant(1, dtype=tf.float32)
                where = tf.not_equal(current_output, zero)

                current_output = tf.clip_by_value(current_output[where],
                                                  clip_value_min=6.755,  # hyper parameter to tune...
                                                  clip_value_max=500)

                where = tf.not_equal(current_output, one)

                current_output = current_output[where]

                total = tf.math.count_nonzero(current_output)

                current_max = tf.math.reduce_max(current_output)  # (tf.keras.backend.sum(current_output)
                               # / tf.cast(total, tf.float32))

                if total > 1:
                    current_max = tf.keras.backend.sum(current_output) / tf.cast(total, tf.float32)

                outputs.append(current_max)
                if VERBOSE:
                    print("\nWorking on file {} out of {}!!!\n"
                          .format(i + 1, files_len))
                    print("file: {}".format(str(md)))
                    print(current_output)
                    print(tf.shape(current_output))
                    print(current_max)

                    if current_max > max_max:
                        max_max = current_max
                        best_candidate = my_file[2][tf.argmax(outputs)]
                    print(max_max)
                    print("Best candidate: {}".format(best_candidate))

    index_max = tf.argmax(outputs)

    with open(corpus_path + "/" + my_file[2][index_max]) as f:
        return f.read()


def text_search(question, reference):
    """"""
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
    """"""
    txt_to_search = semantic_search(reference, question)
    return text_search(question, txt_to_search)


def qa_bot(reference):
    """"""
    while True:
        print("Q: ", end="")
        question = input().lower()

        if ((question == "exit"
             or question == "quit"
             or question == "goodbye"
             or question == "bye")):
            print("A: Goodbye")
            exit()

        answer = answer_question(question, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: ", answer)
