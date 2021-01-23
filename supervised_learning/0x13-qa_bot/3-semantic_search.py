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
    max_max = -999
    best_candidate = None

    for my_file in files:
        files_len = len(my_file[2])
        for i, md in enumerate(my_file[2]):
            with open(corpus_path + "/" + str(md)) as f:
                reference = f.read()

                sentence_tokens = tokenizer.tokenize(sentence)
                paragraph_tokens = tokenizer.tokenize(reference)
                tokens = (["[CLS]"] + sentence_tokens + ["[SEP]"]
                          + paragraph_tokens + ["[SEP]"])

                input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_masks = [1] * len(input_word_ids)
                input_type_ids = ([0] * (1 + len(sentence_tokens) + 1) + [1]
                                  * (len(paragraph_tokens) + 1))

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
                current_max_start = tf.argmax(output[0][0][1:])
                current_max_end = tf.argmax(output[1][0][1:])
                total = current_max_end - current_max_start
                current_output = output[0][0][current_max_start:
                                              current_max_end + 1]

                current_max = (tf.keras.backend.sum(current_output)
                               / tf.cast(total, tf.float32))

                # using `[1:]` will enforce an answer. `outputs[0][0][0]`
                # is the ignored '[CLS]' token logit
                # short_start = tf.argmax(outputs[0][0][1:]) + 1
                # short_end = tf.argmax(outputs[1][0][1:]) + 1
                # answer_tokens.append(tokens[short_start:short_end + 1])
                print(tf.shape(current_output)[0])
                if current_max == 0 or tf.shape(current_output)[0] <= 2:
                    current_max = -999
                outputs.append(current_max)
                if VERBOSE:
                    print("\nWorking on file {} out of {}!!!\n"
                          .format(i + 1, files_len))
                    print("file: {}".format(str(md)))

                    print(current_output)

                    print(current_max)

                    if current_max > max_max:
                        max_max = current_max
                        best_candidate = my_file[2][tf.argmax(outputs)]
                    print(max_max)
                    print("Best candidate: {}".format(best_candidate))

    index_min = tf.argmin(outputs)
    print(my_file[2][index_min])

    index_max = tf.argmax(outputs)
    print(my_file[2][index_max])
    print("\n")

    # return tokenizer.convert_tokens_to_string(answer_tokens[index])
    with open(corpus_path + "/" + my_file[2][index_max]) as f:
        return f.read()
