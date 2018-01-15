# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pykit.utils import to_unicode


def str_half2full(s):
    def _conv(c):
        code = ord(c)
        if code == 0x0020:
            code = 0x3000
        elif 0x21 <= code <= 0x7e:
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


