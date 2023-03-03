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
    print(code)
    target = '''
(defun fn (arg0)
    (let*
        ()
        (substring arg0 2 3)
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
    print(code)

    target = '''
(defun fn (arg0)
    (let*
        ()
        (concat "hello" "world")
    )
)
'''
    assert code.strip() == target.strip()
