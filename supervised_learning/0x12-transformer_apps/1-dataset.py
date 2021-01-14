#!/usr/bin/env python3
""""""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()

class Dataset():
    """"""

    def __init__(self):
        """"""
        ds = tfds.load("ted_hrlr_translate/pt_to_en", split="train", as_supervised=True)
        self.data_train = ds

        ds = tfds.load("ted_hrlr_translate/pt_to_en", split="validation", as_supervised=True)
        self.data_valid = ds

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """"""
        # data.map(lambda self, x: tf.py_function(my_to_string, [x], [tf.string]))

        pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in data), target_vocab_size=2**15)
        en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in data), target_vocab_size=2**15)

        return pt, en

    def encode(self, pt, en):
        """"""
        pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt, en
