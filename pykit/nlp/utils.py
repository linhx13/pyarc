# -*- coding: utf-8 -*-

from __future__ import absolute_import

import re


__english_periods = u'\r|\n|\?!|!|\?|\. '
__three_periods = u'？！”|。’”|！’”|……”'
__two_periods = u'。”|！”|？”|；”|？！|……'
__one_periods = u'！|？|｡|。|'

__periods_pat = re.compile(u'(%s)' % '|'.join(
    [__english_periods, __three_periods, __two_periods, __one_periods]))


def split_sentences(text):
    ''' Split text into sentences.

    Args:
      text: unicode string needed to split
    Returns:
      a list of sentences
    '''
    res = __periods_pat.split(text)
    return (''.join(res[i:i+2]).strip() for i in xrange(0, len(res), 2))


def str_half2full(text):
    ''' Convert text from half-width characters to full-width characters. '''
    def _conv(c):
        code = ord(c)
        if code == 0x0020:
            code = 0x3000
        elif 0x0021 <= code <= 0x007e:
            code += 0xfee0
        return unichr(code)
    return ''.join(_conv(c) for c in text)


def str_full2half(text):
    '''Conver text from full-width characters to full-width characters. '''
    def _conv(c):
        code = ord(c)
        if code == 0x3000:
            code = 0x0020
        elif 0xff01 <= code <= 0xff5e:
            code -= 0xfee0
        return unichr(code)
    return ''.join(_conv(c) for c in text)


FH_SPACE = FHS = ((u"　", u" "),)
FH_NUM = FHN = (
    (u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
    (u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9"),
)
FH_ALPHA = FHA = (
    (u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
    (u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
    (u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
    (u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
    (u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"),
    (u"ｚ", u"z"),
    (u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
    (u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
    (u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
    (u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
    (u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"),
    (u"Ｚ", u"Z"),
)
FH_PUNCTUATION = FHP = (
    (u"．", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"”", u'"'),
    (u"’", u"'"), (u"‘", u"`"), (u"＠", u"@"), (u"＿", u"_"), (u"：", u":"),
    (u"；", u";"), (u"＃", u"#"), (u"＄", u"$"), (u"％", u"%"), (u"＆", u"&"),
    (u"（", u"("), (u"）", u")"), (u"‐", u"-"), (u"＝", u"="), (u"＊", u"*"),
    (u"＋", u"+"), (u"－", u"-"), (u"／", u"/"), (u"＜", u"<"), (u"＞", u">"),
    (u"［", u"["), (u"￥", u"\\"), (u"］", u"]"), (u"＾", u"^"), (u"｛", u"{"),
    (u"｜", u"|"), (u"｝", u"}"), (u"～", u"~"),
)

FH_ASCII = HAC = lambda: ((fr, to) for m in (FH_ALPHA, FH_NUM, FH_PUNCTUATION)
                          for fr, to in m)

HF_SPACE = HFS = ((u" ", u"　"),)
HF_NUM = HFN = lambda: ((h, z) for z, h in FH_NUM)
HF_ALPHA = HFA = lambda: ((h, z) for z, h in FH_ALPHA)
HF_PUNCTUATION = HFP = lambda: ((h, z) for z, h in FH_PUNCTUATION)
HF_ASCII = ZAC = lambda: ((h, z) for z, h in FH_ASCII())


def convert_fh(text, *maps, **ops):
    """ Convert between full-width and half-width characters.

    Args:
      text: unicode string need to convert
      maps: conversion maps
      skip: skip out of character. In a tuple or string
    Returns:
      converted unicode string
    """
    if "skip" in ops:
        skip = ops["skip"]
        if isinstance(skip, basestring):
            skip = tuple(skip)

        def replace(text, fr, to):
            return text if fr in skip else text.replace(fr, to)
    else:
        def replace(text, fr, to):
            return text.replace(fr, to)

    for m in maps:
        if callable(m):
            m = m()
        elif isinstance(m, dict):
            m = m.items()
        for fr, to in m:
            text = replace(text, fr, to)
    return text


if __name__ == '__main__':
    text = u"成田空港—【ＪＲ特急成田エクスプレス号・横浜行，2站】—東京—【ＪＲ新幹線はやぶさ号・新青森行,6站 】—新青森—【ＪＲ特急スーパー白鳥号・函館行，4站 】—函館"
    print(convert_fh(text, FH_ASCII,
                     {u"【": u"[", u"】": u"]", u",": u"，", u".": u"。",
                      u"?": u"？", u"!": u"！"},
                     spit="，。？！“”"))
