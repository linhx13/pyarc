# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
import types

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if not PY2:
    string_types = (str,)
    integer_types = (int,)
    class_types = (type,)
    text_type = str
    binary_type = bytes

    def iterkeys(d, **kwargs):
        return iter(d.keys(**kwargs))

    def itervalues(d, **kwargs):
        return iter(d.values(**kwargs))

    def iteritems(d, **kwargs):
        return iter(d.items(**kwargs))

    from io import StringIO

    xrange = range

else:
    string_types = (basestring,)
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
    binary_type = str

    def iterkeys(d, **kwargs):
        return d.iterkeys(**kwargs)

    def itervalues(d, **kwargs):
        return d.itervalues(**kwargs)

    def iteritems(d, **kwargs):
        return d.iteritems(**kwargs)

    from cStringIO import StringIO
