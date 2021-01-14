#!/usr/bin/env python3
""""""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()


class Dataset():
    """The Dataset class loads and processes data"""

    def __init__(self, batch_size, max_len):
        """Init dataset"""
        dst = tfds.load("ted_hrlr_translate/pt_to_en", split="train",
                        as_supervised=True)
        self.data_train = dst

        dsv = tfds.load("ted_hrlr_translate/pt_to_en", split="validation",
                        as_supervised=True)
        self.data_valid = dsv

        self.tokenizer_pt, self.tokenizer_en = (self
                                                .tokenize_dataset(self
                                                                  .data_train))

    def tokenize_dataset(self, data):
        """Tokenize the data set"""
        pt = (tfds.deprecated.text.SubwordTextEncoder
              .build_from_corpus((pt.numpy() for pt, en in data),
                                 target_vocab_size=2**15))
        en = (tfds.deprecated.text.SubwordTextEncoder
              .build_from_corpus((en.numpy() for pt, en in data),
                                 target_vocab_size=2**15))

        return pt, en
