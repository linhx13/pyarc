# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)


class Tokenizer(object):
    ''' A Tokenizer splits strings into sequences of tokens that can be used in
    a model. The `tokens` here could be words, characters, or words and
    characters.'''
