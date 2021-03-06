# -*- coding: utf-8 -*-


from __future__ import absolute_import

import logging
import re
from collections import Counter
import itertools
import math
from six import string_types

from scipy.stats import entropy
from gensim.utils import SaveLoad
from pyarc.utils import to_utf8, to_unicode

logger = logging.getLogger(__name__)


class WordInfo(object):
    ''' Store information of each word.'''
    def __init__(self, text):
        self.text = text
        self.count = 0
        self.left_neighbours = Counter()
        self.right_neighbours = Counter()

    def __str__(self):
        ''' Get short string representation of this word info. '''
        return '%s<text="%s", count=%d, %d left_neighbours, ' \
            '%d right_neighbours>' % \
            (self.__class__.__name__, to_utf8(self.text), self.count,
             len(self.left_neighbours), len(self.right_neighbours))

    def update(self, left_token, right_token):
        ''' Update word info, and add left/right neightbours. '''
        self.count += 1
        if left_token:
            self.left_neighbours[left_token[-1]] += 1
        if right_token:
            self.right_neighbours[right_token[0]] += 1


class WordDetector(SaveLoad):
    """ Detect words from unlabeled text data. """

    SEP = '$'
    NON_CHINESE_PATTERN = re.compile(u'[^\u4E00-\u9FA50-9]+')
    WHITESPACE_PATTERN = re.compile(u'\s+')

    def __init__(self, sentences=None, max_word_len=5, min_count=5,
                 min_cohension=1.0, min_freedom=0.5, max_vocab_size=40000000,
                 progress_per=10000):
        if max_word_len <= 0:
            raise ValueError('max_word_len should be at least 1')
        if min_count <= 0:
            raise ValueError('min_count should be at least 1')
        if min_cohension < -1 or min_cohension > 1:
            raise ValueError('min_cohension should be between -1 and 1')
        if min_freedom <= 0:
            raise ValueError('min_freedom should be larger than 0')

        self.max_word_len = max_word_len
        self.min_count = min_count
        self.min_cohension = min_cohension
        self.min_freedom = min_freedom
        self.max_vocab_size = max_vocab_size
        self.progress_per = progress_per
        self.corpus_count = 0
        self.vocab = {}

        if sentences is not None:
            self.fit(sentences)

    def __str__(self):
        ''' Get short string representation of this word detector. '''
        return '%s<%d vocab, corpus_count=%d,  min_count=%d, ' \
            'min_cohension=%f, min_freedom=%f, max_vocab_size=%f>' % \
            (self.__class__.__name__, len(self.vocab), self.corpus_count,
             self.min_count, self.min_cohension, self.min_freedom,
             self.max_vocab_size)

    def fit(self, sentences):
        ''' Fit the collected word info into this word detector.

        Args:
          sentences: an iterable of string or token-list.
        '''
        logger.info('Collecting all words and their counts ...')
        sentence_no = -1
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % self.progress_per == 0:
                logger.info('PROGRESS: at sentence %d', sentence_no)
            try:
                sen_len, sentence = \
                    self._preprocess_sentence(sentence, self.SEP)
                self.corpus_count += sen_len
                for suf in self._index_of_suffix(1, len(sentence) - 1,
                                                 self.max_word_len):
                    word = sentence[suf[0]:suf[1]]
                    if word not in self.vocab:
                        self.vocab[word] = WordInfo(word)
                    logger.debug('suf:%s, word:%s, left:%s, right:%s' %
                                 (suf, to_utf8(word),
                                  to_utf8(sentence[suf[0] - 1]),
                                  to_utf8(sentence[suf[1]])))
                    self.vocab[word].update(sentence[suf[0] - 1],
                                            sentence[suf[1]])
            except Exception as ex:
                logger.warn('Processing error: %s', ex)
                continue

    def partial_fit(self, sentences):
        self.fit(sentences)

    @staticmethod
    def _preprocess_sentence(sentence, sep):
        ''' Preprocess sentence.

        Args:
          sentence: the sentence.
          sep: sep to append in front and end of the setence.
        '''
        if isinstance(sentence, list):
            sentence = map(to_unicode, sentence)
            ori_len = len(sentence)
            if sep:
                sentence = list(itertools.chain([sep], sentence, [sep]))
        elif isinstance(sentence, string_types):
            sentence = re.sub(WordDetector.WHITESPACE_PATTERN, ' ',
                              to_unicode(sentence))
            ori_len = len(sentence)
            if sep:
                sentence = '%s%s%s' % (sep, sentence, sep)
        else:
            raise ValueError('sentence should be of list or string_types')
        return ori_len, sentence

    @staticmethod
    def _index_of_suffix(start_idx, end_idx, max_word_len):
        for i in xrange(start_idx, end_idx):
            for j in xrange(i + 1, min(i + 1 + max_word_len, end_idx + 1)):
                yield (i, j)

    @staticmethod
    def _get_subparts(text):
        ''' Partition a text into all possible two parts.

        Args:
          text: could be a string or a tuple.
        '''
        for i in xrange(1, len(text)):
            yield ((text[:i], text[i:]))

    @staticmethod
    def _is_all_non_chinese(text):
        if not isinstance(text, string_types):
            text = ''.join(text)
        m = re.findall(WordDetector.NON_CHINESE_PATTERN, text)
        return len(m) == 1 and m[0] == text

    @staticmethod
    def _has_non_chinese(text):
        if not isinstance(text, string_types):
            text = ''.join(text)
        return re.findall(WordDetector.NON_CHINESE_PATTERN, to_unicode(text))

    def merge_with(self, other):
        ''' Merge word infos of another WordDetector.

        `max_word_len` of two WordDetector should be same.

        Args:
          other: another WordDetector.
        '''
        if self.max_word_len != other.max_word_len:
            raise ValueError('max_word_len of two WordDetector [%d, %d] '
                             'is not same' %
                             (self.max_word_len, other.max_word_len))
        self.corpus_count += other.corpus_count
        for word, other_info in other.vocab.iteritems():
            this_info = self.vocab.get(word, WordInfo(word))
            this_info.count += other_info.count
            this_info.left_neighbours.update(other_info.left_neighbours)
            this_info.right_neighbours.update(other_info.right_neighbours)

    def export_words(self):
        ''' Export an iterator that contains all words according to the params.
        '''
        for word, info in self.vocab.iteritems():
            if isinstance(word, string_types) and (len(word) <= 1):
                continue
            if self._has_non_chinese(word):
                continue
            if self._is_all_non_chinese(word):
                continue
            cohension = self._cohension_scorer(word, self.vocab,
                                               self.corpus_count)
            freedom = self._freedom_scorer(word, self.vocab)
            logger.debug('word:%s, count:%d, cohension:%f, freedom:%f',
                         ''.join(word), info.count, cohension, freedom)
            if info.count >= self.min_count \
               and cohension >= self.min_cohension \
               and freedom >= self.min_freedom:
                total_score = self._total_scorer(word, self.vocab,
                                                 cohension, freedom)
                yield (word, info.count, cohension, freedom, total_score)

    @staticmethod
    def _cohension_scorer(word, vocab, corpus_count):
        info = vocab[word]
        parts = list(WordDetector._get_subparts(word))
        c = min(itertools.imap(lambda p: 1.0 * info.count /
                               (vocab[p[0]].count * vocab[p[1]].count),
                               parts))
        return math.log(c * corpus_count) / \
            -math.log(1.0 * info.count / corpus_count)

    @staticmethod
    def _freedom_scorer(word, vocab):
        info = vocab[word]
        left_entropy = entropy(info.left_neighbours.values())
        right_entropy = entropy(info.right_neighbours.values())
        return min(left_entropy, right_entropy)

    @staticmethod
    def _npmi_scorer(worda_count, wordb_count, wordab_count, corpus_count):
        if corpus_count <= 0:
            raise ValueError('corpus_count should be larger than 0')
        pa = 1.0 * worda_count / corpus_count
        pb = 1.0 * wordb_count / corpus_count
        pab = 1.0 * wordab_count / corpus_count
        return math.log(pab / (pa * pb)) / -math.log(pab)

    @staticmethod
    def _total_scorer(word, vocab, cohension, freedom):
        info = vocab[word]
        return info.count * (cohension + freedom)


if __name__ == '__main__':
    import gc
    gc.disable()
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--max_word_len', type=int, default=5)
    parser.add_argument('--min_count', type=int, default=5)
    parser.add_argument('--min_cohension', type=float, default=0.3)
    parser.add_argument('--min_freedom', type=float, default=1.0)
    parser.add_argument('--progress_per', type=int, default=1000)
    args = parser.parse_args()
    logger.info('args: %s' % args)

    # sentences = ['十四是十四四十是四十，，十四不是四十，，，，四十不是十四']
    # sentences = ['十四是十四，四十是四十']
    # sentences = ['十四是十四']
    # sentences = [['十四', '是', '十四']]
    import sys
    if args.input:
        sentences = open(args.input)
    else:
        sentences = sys.stdin
    wd = WordDetector(sentences, max_word_len=args.max_word_len,
                      min_count=args.min_count,
                      min_cohension=args.min_cohension,
                      min_freedom=args.min_freedom,
                      progress_per=args.progress_per)
    logger.info('wd: %s' % wd)
    from operator import itemgetter
    words = sorted(wd.export_words(), key=itemgetter(4), reverse=True)
    if args.output:
        fout = open(args.output, 'w')
    else:
        fout = sys.stdout
    for word, count, cohension, freedom, score in words:
        fout.write('%s\t%d\t%f\t%f\t%f\n' %
                   (to_utf8(word), count, cohension, freedom, score))
