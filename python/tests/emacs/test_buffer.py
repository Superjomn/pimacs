'''
This file would be translated into a elisp script and run all the tests in emacs.
'''

from pyimacs.aot import AOTFunction
from pyimacs.elisp.buffer import Buffer
from pyimacs.elisp.core import Guard

import pyimacs


@pyimacs.aot
def _test_buffer_basic():
    some_buffer = Buffer("test")
    with Guard("with-current-buffer", some_buffer):
        some_buffer.insert("hello world")
        assert some_buffer.get_content() == "hello world"


@pyimacs.aot
def main():
    _test_buffer_basic()


def test_buffer():
    code = AOTFunction.to_lispcode()
    print(code)


test_buffer()
