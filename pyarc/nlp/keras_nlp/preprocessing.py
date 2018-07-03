# -*- coding: utf-8 -*-

import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_max_shape(data):
    if not hasattr(data, '__iter__'):
        raise ValueError('`data` should be iterable.')
    if len(data) == 0 or not hasattr(data[0], '__iter__'):
        return (len(data),)
    sub_shapes = []
    for sub_data in data:
        sub_shapes.append(get_max_shape(sub_data))
    max_sub_shape = np.max(sub_shapes, axis=0)
    return (len(data),) + tuple(max_sub_shape)


def _pad_sequences(sequences, results, maxlen, dtype='int32', padding='pre',
                   truncating='pre', value=0.0):
    if maxlen is None:
        raise ValueError('`maxlen` shoud not be None.')

    for idx, seq in enumerate(sequences):
        if not len(seq):
            continue

        if truncating == 'pre':
            trunc = seq[-maxlen[0]:]
        elif truncating == 'post':
            trunc = seq[:maxlen[0]]
        else:
            raise ValueError('Truncating type "%s" not understood' %
                             truncating)

        # Apply padding.
        if padding == 'post':
            if hasattr(trunc[0], '__iter__'):
                _pad_sequences(trunc, results[idx, :len(trunc)], maxlen[1:],
                               padding=padding, truncating=truncating)
            else:
                results[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            if hasattr(trunc[0], '__iter__'):
                _pad_sequences(trunc, results[idx, -len(trunc):], maxlen[1:],
                               padding=padding, truncating=truncating)
            else:
                results[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' %
                             padding)
    return results


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.0):
    if isinstance(maxlen, (int, long)):
        maxlen = (maxlen,)
    if maxlen is None or None in maxlen:
        max_shape_computed = get_max_shape(sequences)
        maxlen_computed = max_shape_computed[1:]
        if maxlen is None:
            maxlen = maxlen_computed
        else:
            maxlen = np.max([maxlen, maxlen_computed], axis=0)

    results_shape = (len(sequences),) + tuple(maxlen)
    results = (np.ones(results_shape) * value).astype(dtype)
    return _pad_sequences(sequences, results, maxlen=maxlen, dtype=dtype,
                          padding=padding, truncating=truncating, value=value)
