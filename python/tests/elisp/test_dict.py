import ast

import astpretty
import pytest
from pyimacs.aot import aot, get_context
from pyimacs.elisp.dict import Dict
from pyimacs.lang import ir

from pyimacs import compile

ctx = get_context()


def test_dict_basic():
    @aot
    def fn():
        d = Dict()
        d["hello"] = "world"
        return d["hello"]

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compile(fn, builder=builder, module=module)
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
