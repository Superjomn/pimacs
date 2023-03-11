import pytest
from pyimacs.aot import aot, get_context
from pyimacs.compiler import compile
from pyimacs.elisp.string import String
from pyimacs.lang import ir

from pyimacs import compile

ctx = get_context()


def test_string_substring():
    @aot
    def fn(name: str) -> str:
        s = String(name)
        s = s[2:3]
        return s

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compile(fn, builder=builder, module=module)
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

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compile(fn, builder=builder, module=module)
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
