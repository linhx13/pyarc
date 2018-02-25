# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function


import codecs
from six import string_types
from pykit.utils import to_unicode


class MMSegmentor(object):
    def __init__(self, word_set):
        if isinstance(word_set, string_types):
            with codecs.open(word_set, 'rb', 'utf-8') as fin:
                self.word_set = set(x.strip() for x in fin)
        elif isinstance(word_set, (dict, set)):
            self.word_set = word_set
        else:
            raise TypeError('error type word_set')

    def fmm_cut(self, sentence, max_len=7):
        ''' Forward maximum matching. '''
        sentence = to_unicode(sentence)
        sen_len = len(sentence)
        cur = 0
        res = []
        while cur < sen_len:
            for i in xrange(max_len, 0, -1):
                if sentence[cur:cur+i] in self.word_set:
                    break
            res.append(sentence[cur:cur+i])
            cur += i
        return res

    def rmm_cut(self, sentence, max_len=7):
        ''' Reverse maximum matching. '''
        sentence = to_unicode(sentence)
        sen_len = len(sentence)
        res = []
        cur = sen_len
        while cur > 0:
            if max_len > cur:
                max_len = cur
            for i in xrange(max_len, 0, -1):
                if sentence[cur-i:cur] in self.word_set:
                    break
            res.append(sentence[cur-i:cur])
            cur -= i
        return res[::-1]

    def bimm_cut(self, sentence, max_len=7):
        ''' Bi-direction maximum matching. '''
        pass


if __name__ == '__main__':
    text = u'我爱北京天安门'
    word_dict = {u"爱":1,u"北京":1,u"天安门":1,u"研究":1,u"研究生":1,u"中国":1,u"国人":1,u"一件":1,u"面子":1,u"一个":1}
    seg = MMSegmentor(word_dict)
    for t in seg.fmm_cut(text):
        print(t)
    for t in seg.rmm_cut(text):
        print(t)

