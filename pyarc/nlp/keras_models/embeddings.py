# -*- coding: utf-8 -*-

from __future__ import absolute_import

import logging
import itertools
import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


def build_embedding_index_from_w2v(fname, binary=True):
    ''' Build token embedding index from word2vec model. '''
    w2v = KeyedVectors.load_word2vec_format(fname, unicode_errors='ignore',
                                            binary=binary)
    return dict(itertools.izip(w2v.index2word, w2v.syn0))


def build_embedding_weights(token_index, embedding_index):
    ''' Buld an embedding matrix for all tokens in token_index with
    embedding_index.
    '''
    if len(embedding_index) == 0:
        raise ValueError('embeddign_index size cannot be 0')
    embedding_dim = embedding_index.itervalues().next().shape[-1]
    embedding_weights = np.zeros((len(token_index) + 1, embedding_dim))
    for token, idx in token_index.iteritems():
        vec = embedding_index.get(token)
        if vec is not None:
            embedding_weights[idx] = vec
    return embedding_weights
