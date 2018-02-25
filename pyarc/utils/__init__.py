# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import with_statement


import logging
from contextlib import contextmanager
import numpy as np
import os
import sys
from collections import defaultdict
from itertools import izip
import random

import six
from six.moves import xrange
from six import iteritems


if not six.PY2:
    unicode = str


logger = logging.getLogger(__name__)


def to_utf8(text, encoding='utf-8', errors='strict'):
    """Convert a string (unicode or bytestring in `encoding`), to bystring in
    utf-8 encoding. """
    if isinstance(text, unicode):
        return text.encode('utf-8')
    else:
        return unicode(text, encoding, errors=errors).encode('utf-8')


def to_unicode(text, encoding='utf-8', errors='strict'):
    """Convert a string (bytesstring in `encoding` or unicode) to unicode. """
    if isinstance(text, unicode):
        return text
    else:
        return unicode(text, encoding, errors=errors)


def codecs_open(filename, mode='r', encoding=None, errors=None):
    if six.PY2:
        import codecs
        return codecs.open(filename, mode=mode, encoding=encoding,
                           errors=errors)
    else:
        return open(filename, mode=mode, encoding=encoding, errors=errors)


@contextmanager
def file_or_filename(input):
    """Return a file-like object ready to be read from the beginning.

    Args:
        input: either a filename or a file-like object supporting seeking
    """
    if isinstance(input, six.string_types):
        # input is a filename, open as a file
        yield open(input)
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        yield input


def load_keyed_vector_iter(filename, encoding='utf-8', kv_sep='\t',
                           vec_sep=',', key_func=None):
    """Load keyed vector iter from a file.

    Args:
      filename: the file to load data from.
        Each line format: <label>: <vec>

    Returns:
      A vector iterator over (label, vec).
    """
    with codecs_open(filename, encoding=encoding) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            key, vec_str = line.split(kv_sep, 1)
            if key_func is not None:
                key = key_func(key)
            try:
                vec = [float(i) for i in vec_str.split(vec_sep)
                       if i.strip() != '']
                dim = len(vec)
            except Exception as ex:
                logging.warn('Loading vec error: %s, input vec line: %s' %
                             (ex, to_utf8('utf-8')))
                vec = [0.0] * dim
            vec = np.array(vec)
            yield key, vec


def dump_keyed_vector_iter(vec_iter, filename, encoding='utf-8', kv_sep='\t',
                           vec_sep=',', key_func=None):
    with codecs_open(filename, 'wb', encoding=encoding) as fout:
        for key, vec in vec_iter:
            if key_func is not None:
                key = key_func(key)
            fout.write('%s%s%s\n' %
                       (key, kv_sep, vec_sep.join(map(to_unicode, vec))))


def load_keyed_vector_dict(filename, encoding='utf-8', kv_sep='\t',
                           vec_sep=',', key_func=None):
    return dict(load_keyed_vector_iter(filename, encoding, kv_sep, vec_sep,
                                       key_func))


def dump_keyed_vector_dict(dict_obj, filename, encoding='utf-8', kv_sep='\t',
                           vec_sep=',', key_func=None):
    return dump_keyed_vector_iter(iteritems(dict_obj), filename, encoding,
                                  kv_sep, vec_sep, key_func)


def load_kv_iter(filename, encoding='utf-8', kv_sep='\t', key_func=None,
                 value_func=None):
    with codecs_open(filename, encoding=encoding) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(kv_sep, 1)
            if key_func is not None:
                key = key_func(key)
            if value_func is not None:
                value = value_func(value)
            yield key, value


def dump_kv_iter(kv_iter, filename, encoding='utf-8', kv_sep='\t',
                 key_func=None, value_func=None):
    with codecs_open(filename, 'w', encoding) as fout:
        for key, value in kv_iter:
            if key_func is not None:
                key = key_func(key)
            if value_func is not None:
                value = value_func(value)
            fout.write('%s%s%s\n' % (key, kv_sep, value))


def load_dict(filename, encoding='utf-8', kv_sep='\t', key_func=None,
              value_func=None):
    return dict(load_kv_iter(filename, encoding, kv_sep, key_func, value_func))


def dump_dict(dict_obj, filename, encoding='utf-8', kv_sep='\t',
              key_func=None, value_func=None):
    dump_kv_iter(iteritems(dict_obj), filename, encoding, kv_sep, key_func,
                 value_func)


def memory_usage():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(2 ** 20)
    except ImportError:
        import resource
        rusage_denom = 1024.
        if sys.platform == 'darwin':
            # ... it seems that in OSX the output is different units ...
            rusage_denom = rusage_denom * rusage_denom
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem


def namedtuple_with_defaults(typename, field_names, default_values=()):
    from collections import namedtuple, Mapping
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def convert_to_defaultdict(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        k = defaultdict(int, izip(xrange(len(x)), x))
    elif isinstance(x, dict):
        k = defaultdict(int, x)
    elif isinstance(x, set):
        k = defaultdict.fromkeys(x, 1)
        k.default_factory = int
    else:
        raise ValueError("Invalid param type %s for jaccard similarity" %
                         type(x))
    return k


def random_sample(input_data, sample_count, seed=None):
    random.seed(seed)
    output_data = []
    for data_idx, data in enumerate(input_data):
        if len(output_data) < sample_count:
            output_data.append(data)
            continue
        r = random.randint(0, data_idx + 1)
        if r < len(output_data):
            output_data[r] = data
    return output_data
