import ast

import astpretty
import pytest
from pyimacs.elisp.dict import Dict
from pyimacs.runtime import aot

from pyimacs import compile


def test_dict_basic():
    @aot
    def fn():
        d = Dict()
        d["hello"] = "world"
        return d["hello"]

    code = compile(fn, signature="void->s")
    target = '''
(defun fn ()
    (let*
        (arg0)
        (setq arg0 (ht-create))
        (ht-set! arg0 "hello" "world")
        (ht-get arg0 "hello")
    )
)
    '''
    print(code)
    assert code.strip() == target.strip()
