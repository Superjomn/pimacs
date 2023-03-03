import pytest
from pyimacs.elisp.string import String
from pyimacs.runtime import jit

from pyimacs import compile


def test_string_substring():
    @jit
    def fn(name):
        s = String(name)
        s = s[2:3]
        return s
    code = compile(fn, signature="s->s")
    target = '''
(defun fn (arg0)
    (let*
        (arg1 arg2 arg3)
        (setq arg1 2)
        (setq arg2 3)
        (setq arg3 (substring arg0 arg1 arg2))
        arg3
    )
)
    '''
    assert code.strip() == target.strip()


def test_string_concat():
    @jit
    def fn():
        a = String("hello")
        b = String("world")

        return a + b

    code = compile(fn, signature="void->s")

    target = '''
(defun fn (arg0)
    (let*
        (arg1 arg2 arg3)
        (setq arg1 "hello")
        (setq arg2 "world")
        (setq arg3 (concat arg1 arg2))
        arg3
    )
)
'''
    assert code.strip() == target.strip()
