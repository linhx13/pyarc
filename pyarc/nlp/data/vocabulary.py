# -*- coding: utf-8 -*-


from __future__ import with_statement

import logging
import itertools
import utils
from collections import Counter


logger = logging.getLogger(__name__)

try:
    import numpy as np

    def calc_idf_iter(tokenid_df_iter, n_docs, smooth_idf):
        tokenid_df_list = list(tokenid_df_iter)
        df_arr = np.array([freq for tokenid, freq in tokenid_df_list],
                          dtype='float')
        idf_arr = \
            np.log(float(n_docs + smooth_idf) / (df_arr + smooth_idf)) + 1.0
        return itertools.izip((x[0] for x in tokenid_df_list), idf_arr)
except ImportError:
    import math

    def calc_idf(tokenid_df_iter, n_docs, smooth_idf=True):
        def _idf(doc_freq):
            return math.log(float(n_docs + smooth_idf) /
                            (doc_freq + smooth_idf)) + 1.0
        return itertools.imap(lambda x: (x[0], _idf(x[1])), tokenid_df_iter)


class Vocabulary(object):
    """ Vocabulary encapsulates the mapping between words and interger ids.
    """
    def __init__(self, documents=None, prune_at=None):
        self.token2id = {}  # token -> token_id
        self.id2token = {}  # token_id -> token, only formed on request
        self.df = {}  # token_id -> document frequency
        self.idf = {}  # token_id -> inverse document frequency
        self.num_docs = 0  # number of documents processed
        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)

    def __len__(self):
        """ Return the number of token->id mapping in the vocabulary. """
        return len(self.token2id)

    def __str__(self):
        some_keys = list(itertools.islice(self.token2id.iterkeys(), 5))
        return "Vocabulary(%d unique tokens: %s%s)" % \
            (len(self), some_keys, "..." if len(self) > 5 else '')

    def __getitem__(self, tokenid):
        if len(self.id2token) != len(self.token2id):
            # the token -> id mapping has changed
            # recompute id -> token accordingly
            self.id2token = dict((v, k) for k, v in self.token2id.iteritems())
        return self.id2token[tokenid]

    def __iter__(self):
        return self.token2id.itervalues()

    iterkeys = __iter__

    def keys(self):
        return list(self.token2id.itervalues())

    def itervalues(self):
        return self.token2id.iterkeys()

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        for key, value in self.token2id.iteritems():
            yield value, key

    def items(self):
        return list(self.iteritems())

    @staticmethod
    def from_documents(documents):
        return Vocabulary(documents=documents)

    def add_documents(self, docs_iter, prune_at=None):
        """ Update vocabulary from a collection of documents. Each document is
        a list of tokens = **tokenized and normalized** strings (either utf-8
        or unicode).
        """
        for docno, doc in enumerate(docs_iter):
            if docno % 10000 == 0:
                if prune_at is not None and len(self) > prune_at:
                    self.filter_extremes(min_df=0, max_df=1.0, kee_n=prune_at)
                logger.info("Adding document #%d to %s", docno, self)
            self.add_document(doc, update_idf=False)
        self.update_idf()
        logger.info("Built %s from %d documents", self, self.num_docs)

    def add_document(self, document, update_idf=True):
        counter = Counter(w if isinstance(w, unicode) else unicode(w, 'utf-8')
                          for w in document)
        for token, freq in counter.iteritems():
            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)
            tokenid = self.token2id[token]
            self.df[tokenid] = self.df.get(tokenid, 0) + 1
        self.num_docs += 1
        if update_idf:
            self.update_idf()

    def filter_extremes(self, min_df=1, max_df=0.5, keep_tokens=None,
                        keep_n=None):
        """ Filter out tokens that appear in
        1. less then `min_df` documents (absolute number) or
        2. more then `max_df` documents (fraction of total corpus size, *not*
           absolute number).
        3. if tokens are give in keep_tokens (list of strings), they will be
           kept regardless of the `min_df` and `max_df` settings
        4. after (1), (2) and (3), keep only the first `keep_n` most frequent
           tokens (or keep all if `None`).

        After the pruning, shrink resulting gaps in word ids.
        """
        max_df_abs = int(max_df * self.num_docs)

        keep_ids = set()
        if keep_tokens:
            keep_ids = set(self.token2id[v]
                           for v in keep_tokens if v in self.token2id)
        good_ids = (v for v in self.token2id.itervalues()
                    if min_df <= self.df.get(v, 0) <= max_df_abs
                    or v in keep_ids)
        good_ids = sorted(good_ids, key=self.df.get, reverse=True)
        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        bad_tokens = [(self[id], self.df.get(id, 0))
                      for id in set(self).difference(good_ids)]
        logger.info("Discarding %d tokens: %s...", len(self) - len(good_ids),
                    bad_tokens[:10])
        logger.info("Keeping %d tokens which were in no less than %d and "
                    "no more then %d (=%.1f%%) documents",
                    len(good_ids), min_df, max_df_abs, 100.0 * max_df)

        # do the actual filtering, then rebuild dictionary to remove the gaps
        self.filter_tokens(good_ids=good_ids)
        logger.info("Resulting vocabulary: %s", self)

    def filter_n_most_frequent(self, remove_n):
        """ Filter out the `remove_n` most frequent tokens that appear in the
        documents.

        After the pruning, shrink resulting gaps in the token ids.
        """
        most_frequent_ids = sorted((v for v in self.token2id.itervalues()),
                                   key=self.df.get, reverse=True)[:remove_n]
        most_frequent_tokens = [(self[id], self.df.get(id, 0))
                                for id in most_frequent_ids]
        logging.info("Discarding %d tokens: %s ...",
                     len(most_frequent_ids), most_frequent_tokens[:10])
        self.filter_tokens(bad_ids=most_frequent_ids)
        logger.info("Resulting vocabulary: %s", self)

    def filter_tokens(self, bad_ids=None, good_ids=None):
        """
        Remove the selected `bad_ids` tokens from all vocabulary mappings,
        or, keep selected `good_ids` in the mapping and remove the rest.
        """
        if bad_ids is not None:
            bad_ids = set(bad_ids)
            self.token2id = \
                dict((token, tokenid)
                     for token, tokenid in self.token2id.iteritems()
                     if tokenid not in bad_ids)
            self.df = dict((tokenid, freq)
                           for tokenid, freq in self.df.iteritems()
                           if tokenid not in bad_ids)
            self.idf = dict((tokenid, idf)
                            for tokenid, idf in self.idf.iteritems()
                            if tokenid not in bad_ids)
        if good_ids is not None:
            good_ids = set(good_ids)
            self.token2id = \
                dict((token, tokenid)
                     for token, tokenid in self.token2id.iteritems()
                     if tokenid in good_ids)
            self.df = dict((tokenid, freq)
                           for tokenid, freq in self.df.iteritems()
                           if tokenid in good_ids)
            self.idf = dict((tokenid, idf)
                            for tokenid, idf in self.idf.iteritems()
                            if tokenid in good_ids)
        self.compactify()

    def compactify(self):
        """ Assign new token ids to all tokens. """
        logger.debug("Rebuilding vocabulary, shrinking gaps")

        idmap = dict(itertools.izip(self.token2id.itervalues(),
                                    xrange(len(self.token2id))))
        self.token2id = dict((token, idmap[tokenid])
                             for token, tokenid in self.token2id.iteritems())
        self.id2token = {}
        self.df = dict((idmap[tokenid], freq)
                       for tokenid, freq in self.df.iteritems())
        self.idf = dict((idmap[tokenid], idf)
                        for tokenid, idf in self.idf.iteritems())

    def merge_with(self, other):
        """ Merge another vocabulary into this vocabulary, mapping same tokens
        to the same ids and new tokens to new ids.

        `other` can be any id->token mapping (a dict, a Vocabulary, ...).
        """
        for other_id, other_token in other.iteritems():
            if other_token in self.token2id:
                new_id = self.token2id[other_token]
            else:
                new_id = len(self.token2id)
                self.token2id[other_token] = new_id
                self.df[new_id] = 0
            try:
                self.df[new_id] += other.df[other_id]
            except:
                # `other` isn't a Vocabulary (probably just a dict), ignore
                pass
        try:
            self.num_docs += other.num_docs
        except:
            pass
        self.update_idf()

    def update_idf(self):
        self.idf = dict(calc_idf_iter(self.df.iteritems(), self.num_docs,
                                      smooth_idf=True))

    def save_as_text(self, fname, sorted_by_word=True):
        """ Save this Vocabulary to a text file,
        first line: `num_docs:<num_docs>`
        following lines in format:
        `token_id[TAB]token[TAB]document frequency[NEWLINE]`.
        Sorted by token, or by descreasing token document frequency.
        """
        logger.info("Saving vocabulary mapping to %s", fname)
        with utils.open(fname, 'wb', encoding='utf-8') as fout:
            fout.write('num_docs:%d\n' % self.num_docs)
            if sorted_by_word:
                for token, tokenid in sorted(self.token2id.iteritems()):
                    line = "%d\t%s\t%d\n" % (tokenid, token,
                                             self.df.get(tokenid, 0))
                    fout.write(line)
            else:
                for tokenid, freq in sorted(self.df.iteritems(),
                                            key=lambda item: -item[1]):
                    line = "%d\t%s\t%d\n" % (tokenid, self[tokenid], freq)
                    fout.write(line)

    @staticmethod
    def load_from_text(fname):
        """ Load a previously sotred Vocabulary from a text file. """
        result = Vocabulary()
        with utils.codecs_open(fname, encoding='utf-8') as fp:
            result.num_docs = int(fp.readline().strip().split(':')[1])
            for lineno, line in enumerate(fp):
                line = line.strip()
                if not line:
                    continue
                try:
                    tokenid, token, docfreq = line.split('\t')
                except Exception:
                    raise ValueError("Invalid line in vocabulary file %s: %s"
                                     % (fname, line))
                tokenid = int(tokenid)
                if token in result.token2id:
                    raise KeyError("Token %s is defined as ID %d and as ID %d"
                                   % (token, tokenid, result.token2id[token]))
                result.token2id[token] = tokenid
                result.df[token] = int(docfreq)
        result.update_idf()
        return result


class HashVocabulary(dict):
    """ HashVocabulary encapsulates the mapping between normalized token and
    their integer ids.
    """
    pass
