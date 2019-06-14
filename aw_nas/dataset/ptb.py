# -*- coding: utf-8 -*-

import os
import random
from collections import Counter

import torch

from aw_nas.dataset.base import BaseDataset
from aw_nas.utils.exception import expect

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.vocabulary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        expect(os.path.exists(path))
        # Add words to the dictionary
        with open(path, "r", encoding="utf-8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.vocabulary.add_word(word)

        # Tokenize file content
        with open(path, "r", encoding="utf-8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    ids[token] = self.vocabulary.word2idx[word]
                    token += 1

        return ids

class SentenceCorpus(object):
    def __init__(self, path):
        self.vocabulary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        """Tokenizes a text file."""
        expect(os.path.exists(path))
        # Add words to the dictionary
        with open(path, "r", encoding="utf-8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ["<eos>"]
                tokens += len(words)
                for word in words:
                    self.vocabulary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ["<eos>"]
                sent = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    sent[i] = self.vocabulary.word2idx[word]
                sents.append(sent)

        return sents

class PTB(BaseDataset):
    NAME = "ptb"

    def __init__(self, shuffle=False):
        super(PTB, self).__init__()

        self.corpus = SentenceCorpus(self.data_dir)
        self.vocabulary = self.corpus.vocabulary
        self.vocab_size = len(self.vocabulary)

        # merge train/val to train
        train_val = self.corpus.train + self.corpus.valid
        # temp: for debug
        # train_val = train_val[:300]

        if shuffle: # shuffle sentences
            random.shuffle(train_val)

        self.datasets = {
            "train": train_val,
            "ori_train": self.corpus.train,
            "ori_valid": self.corpus.valid,
            "test": self.corpus.test
        }

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "sequence"
