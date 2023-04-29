import logging

import pyimacs.lang as pyl
from pyimacs.aot import AOTFunction, aot, get_context
from pyimacs.elisp.buffer import Buffer
from pyimacs.lang import ir

from pyimacs import compiler


def test_aot():
    ''' Multiple aot function. '''
    @aot
    def fn0(a: int, b: int) -> int:
        return a + b

    @aot
    def fn1(a: int, b: int) -> int:
        return a-b

    code = AOTFunction.to_lispcode()

    target = '''
(defun fn0 (arg0 arg1)
    (let*
        ()
        (+ arg0 arg0)
    )
)


(defun fn1 (arg0 arg1)
    (let*
        ()
        (- arg0 arg0)
    )
)
'''
    print(code)
    assert target.strip() in code.strip()
