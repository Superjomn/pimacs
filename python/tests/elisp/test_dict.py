import ast

import astpretty
import pytest
from pyimacs.elisp.dict import Dict
from pyimacs.runtime import jit

from pyimacs import compile


def test_dict_basic():
    @jit
    def fn():
        d = Dict()
        d["hello"] = "world"
        return d["hello"]

    code = compile(fn, signature="void->s")
    target = '''
(defun fn (arg0)
    (let*
        (arg1)
        (setq arg1 (ht-create))
        (ht-set! arg1 "hello" "world")
        (ht-get arg1 "hello")
    )
)
    '''
    print(code)
    assert code.strip() == target.strip()
