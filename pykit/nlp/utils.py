# -*- coding: utf-8 -*-

from __future__ import absolute_import

import re
from pykit.utils import to_unicode


def str_half2full(s):
    def _conv(c):
        code = ord(c)
        if code == 0x0020:
            code = 0x3000
        elif 0x0021 <= code <= 0x007e:
            code += 0xfee0
        return unichr(code)
    return ''.join(_conv(c) for c in to_unicode(s))


def str_full2half(s):
    def _conv(c):
        code = ord(c)
        if code == 0x3000:
            code = 0x0020
        elif 0xff01 <= code <= 0xff5e:
            code -= 0xfee0
        return unichr(code)
    return ''.join(_conv(c) for c in to_unicode(s))


__english_periods = u'\r|\n|\?!|!|\?|\. '
__three_periods = u'？！”|。’”|！’”|……”'
__two_periods = u'。”|！”|？”|；”|？！|……'
__one_periods = u'！|？|｡|。|'

__periods_pat = re.compile(u'(%s)' % '|'.join(
    [__english_periods, __three_periods, __two_periods, __one_periods]))


def split_sentences(s):
    res = __periods_pat.split(to_unicode(s))[:-1]
    return (''.join(res[i:i+2]).strip() for i in xrange(0, len(res), 2))

