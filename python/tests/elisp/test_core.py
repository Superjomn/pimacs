import ast

import astpretty
import pytest
from pyimacs.aot import aot, get_context
from pyimacs.elisp.buffer import (buffer_size, goto_char, point, point_max,
                                  point_min)
from pyimacs.elisp.core import Guard
from pyimacs.lang import ir

from pyimacs import compile

ctx = get_context()


def test_guard_basic():
    @aot
    def fn() -> int:
        with Guard("with-current-buffer"):
            the_point = point()
            goto_char(the_point + 10)
            return 0

    code = compile(fn)
    print(code)
    target = '''
(defun fn ()
    (let*
        ()
        (with-current-buffer
            (let*
                ()
                (goto_char (+ (point) 10))
                0
            )
        )
    )
)
    '''
    assert target.strip() in code
