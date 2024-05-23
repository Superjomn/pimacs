import pytest

import pimacs.ast.ast as ast
import pimacs.ast.type as _ty
from pimacs.sema.func import *
from pimacs.sema.utils import *


def test_func_symbol():
    symbol = FuncSymbol("foo")
    assert symbol.name == "foo"
    assert symbol.context == ()

    bar = ModuleId("bar")

    symbol = FuncSymbol("foo", (bar,))
    assert symbol.name == "foo"
    assert symbol.context == (bar,)

    baz = ClassId("baz")
    symbol = FuncSymbol("foo", (bar, baz))
    assert symbol.name == "foo"
    assert symbol.context == (bar, baz)

    with pytest.raises(ValueError):
        FuncSymbol("foo", (baz, bar))


def test_func_sig():
    func = Function(name="foo", args=[], return_type=None, loc=None, body=[])
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert not sig.input_types
    assert sig.output_type == _ty.Nil

    # foo(x: Int) -> nil
    func = Function(
        name="foo",
        args=[ast.Arg(name="x", type=_ty.Int, loc=None)],
        return_type=None,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert sig.input_types == (
        (
            "x",
            _ty.Int,
        ),
    )
    assert sig.output_type == _ty.Nil

    # foo(x: Int, y: Int) -> Int
    func = Function(
        name="foo",
        args=[ast.Arg(name="x", type=_ty.Int, loc=None)],
        return_type=_ty.Int,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert sig.input_types == (
        (
            "x",
            _ty.Int,
        ),
    )
    assert sig.output_type == _ty.Int

    # foo(x: Int, y: Int) -> Int
    func = Function(
        name="foo",
        args=[
            ast.Arg(name="x", type=_ty.Int, loc=None),
            ast.Arg(name="y", type=_ty.Int, loc=None),
        ],
        return_type=_ty.Int,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert sig.input_types == (
        ("x", _ty.Int),
        ("y", _ty.Int),
    )
    assert sig.output_type == _ty.Int
