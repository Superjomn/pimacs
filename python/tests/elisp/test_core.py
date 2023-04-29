import ast

import astpretty
import pytest
from pyimacs.aot import aot, get_context
from pyimacs.elisp.buffer import (_buffer_size, _goto_char, _point, _point_max,
                                  _point_min)
from pyimacs.elisp.core import Guard
from pyimacs.lang import ir

from pyimacs import compile

ctx = get_context()


def test_guard_basic():
    @aot
    def fn() -> int:
        with Guard("with-current-buffer"):
            the_point = _point()
            _goto_char(the_point + 10)
            return 0

    code = compile(fn)
    print('code', code)
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
