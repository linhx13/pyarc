# -*- coding: utf-8 -*-

import unittest
from collections import Counter

import torchtext_ext


class TestKerasVocab(unittest.TestCase):
    def test_basic(self):
        counter = Counter({"a": 10})
        vocab = torchtext_ext.KerasVocab(counter)
        print vocab.stoi
        self.assertEqual(vocab.stoi['a'], 1)
        self.assertRaises(KeyError, lambda: vocab.stoi['x'])

        vocab = torchtext_ext.KerasVocab(unk_token='<unk>', counter=counter,
                                         specials=['<unk>', '<pad>'])
        print vocab.stoi
        self.assertEqual(vocab.stoi['a'], 2)
        self.assertEqual(vocab.stoi['x'], vocab.stoi['<unk>'])


class TestKerasField(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
