from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.lang.core import *

from .utility import build_mod0


def test_mod_get_function():
    mod, ctx = build_mod0()
    hello_fn = mod.get_function("add")
    print(hello_fn)
    print('body region', hello_fn.body())
    assert hello_fn


def test_mod_get_function_names():
    mod, ctx = build_mod0()
    funcs = mod.get_function_names()
    assert "add" in funcs


def test_function_get_body():
    mod, ctx = build_mod0()
    hello_fn = mod.get_function("add")
    region = hello_fn.body()
    assert region.size() == 1
    block = region.blocks(0)
    assert block.get_num_arguments() == 1
    ops = block.operations()
    assert len(ops) == 3
    assert ops[0].name() == "arith.constant"
