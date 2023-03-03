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
        (arg1 arg2 arg3 arg4 arg5)
        (setq arg1 (ht-create))
        (setq arg2 "world")
        (setq arg3 "hello")
        (ht-set! arg1 arg3 arg2)
        (setq arg4 "hello")
        (setq arg5 (ht-get arg1 arg4))
        arg5
    )
)
    '''
    assert code.strip() == target.strip()
