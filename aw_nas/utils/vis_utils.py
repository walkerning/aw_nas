# -*- coding: utf-8 -

from __future__ import print_function

from functools import wraps

def _none_proxy(*args, **kwargs):
    return None

class WrapWriter(object):
    def __init__(self, writer, prefix=""):
        self.writer = writer
        self.prefix = prefix + ("/" if prefix and not prefix.endswith("/") else "")

    def is_none(self):
        return self.writer is None

    def get_sub_writer(self, name):
        return WrapWriter(self.writer, prefix=self.prefix+name)

    def __getattr__(self, name):
        if name.startswith("add_"):
            if self.writer is None:
                return _none_proxy
            method = getattr(self.writer, name)
            @wraps(method)
            def _func(tag, *args, **kwargs):
                return method(self.prefix + tag, *args, **kwargs)
            _func.__name__ = method.__name__ # for python 2
            _func.__doc__ = method.__doc__   # for python 2
            return _func
        raise AttributeError()
