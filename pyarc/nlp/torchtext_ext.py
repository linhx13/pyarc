# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import logging
from collections import Counter, OrderedDict
from itertools import chain

import torch
import torchtext
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


class Word2Vec(torchtext.vocab.Vectors):
    def __init__(self, name, binary=True, **kwargs):
        self.binary = binary
        super(Word2Vec, self).__init__(name, **kwargs)

    def cache(self, name, cache, url=None):
        wv = KeyedVectors.load_word2vec_format(name, binary=self.binary)
        self.dim = wv.vector_size
        self.itos = wv.index2word
        self.stoi = {word: i for i, word in enumerate(wv.index2word)}
        self.vectors = torch.Tensor(wv.syn0).view(-1, self.dim)


class DefautKeyDict(dict):
    def __init__(self, default_key=None, *args, **kwargs):
        self.default_key = default_key
        super(DefautKeyDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        try:
            return self[self.default_key]
        except:
            raise KeyError(key)

    def __eq__(self, other):
        if self.default_key != other.default_key:
            return False
        return super(DefautKeyDict, self).__eq__(other)


class KerasVocab(torchtext.vocab.Vocab):
    def __init__(self, counter, unk_token=None, *args, **kwargs):
        super(KerasVocab, self).__init__(counter, *args, **kwargs)
        self.unk_token = unk_token
        stoi = DefautKeyDict(self.unk_token)
        stoi.update(self.stoi)
        self.stoi = stoi


class KerasField(torchtext.data.Field):

    vocab_cls = KerasVocab

    def build_vocab(self, *args, **kwargs):
        """
        Copy from torchtext.data.Field.build_vocab(), except the order of
        Vocab's specials order.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, torchtext.data.Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.pad_token, self.unk_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


class KerasNestedField(torchtext.data.NestedField):
    vocab_cls = KerasVocab
