#!/usr/bin/env python3
"""Dataset to be used with Transformer"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """The Dataset class loads and processes data"""

    def __init__(self):
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
        pt = (tfds.features.text.SubwordTextEncoder
              .build_from_corpus((pt.numpy() for pt, en in data),
                                 target_vocab_size=2**15))
        en = (tfds.features.text.SubwordTextEncoder
              .build_from_corpus((en.numpy() for pt, en in data),
                                 target_vocab_size=2**15))

        return pt, en

    def encode(self, pt, en):
        """encodes the data set"""
        pt = ([self.tokenizer_pt.vocab_size] + self
              .tokenizer_pt.encode(pt.numpy()) + [self
                                                  .tokenizer_pt.vocab_size + 1]
              )
        en = ([self.tokenizer_en.vocab_size] + self
              .tokenizer_en.encode(en.numpy()) + [self
                                                  .tokenizer_en.vocab_size + 1]
              )

        return pt, en
