import pytest
from pyimacs.elisp.string import String
from pyimacs.runtime import aot

from pyimacs import compile


def test_string_substring():
    @aot
    def fn(name: str) -> str:
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
    @aot
    def fn() -> str:
        a = String("hello")
        b = String("world")

        return a + b

    code = compile(fn, signature="void->s")
    print(code)

    target = '''
(defun fn ()
    (let*
        ()
        (concat "hello" "world")
    )
)
'''
    assert code.strip() == target.strip()
