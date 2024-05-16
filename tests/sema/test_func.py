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
    func = FuncDecl(name="foo", args=[], return_type=None, loc=None, body=[])
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert not sig.input_types
    assert sig.output_type == _ty.Nil

    # foo(x: Int) -> nil
    func = FuncDecl(
        name="foo",
        args=[ast.ArgDecl(name="x", type=_ty.Int, loc=None)],
        return_type=None,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert sig.input_types == (_ty.Int,)
    assert sig.output_type == _ty.Nil

    # foo(x: Int, y: Int) -> Int
    func = FuncDecl(
        name="foo",
        args=[ast.ArgDecl(name="x", type=_ty.Int, loc=None)],
        return_type=_ty.Int,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert sig.input_types == (_ty.Int,)
    assert sig.output_type == _ty.Int

    # foo(x: Int, y: Int) -> Int
    func = FuncDecl(
        name="foo",
        args=[
            ast.ArgDecl(name="x", type=_ty.Int, loc=None),
            ast.ArgDecl(name="y", type=_ty.Int, loc=None),
        ],
        return_type=_ty.Int,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert sig.input_types == (
        _ty.Int,
        _ty.Int,
    )
    assert sig.output_type == _ty.Int


def test_func_table():
    table = FuncTable()
    assert not table.lookup("foo")
    assert not table.lookup("bar")

    foo = FuncDecl(name="foo", args=[], return_type=None, loc=None, body=[])
    bar = FuncDecl(name="bar", args=[], return_type=None, loc=None, body=[])
    goo = FuncDecl(name="goo", args=[], return_type=None, loc=None, body=[])

    bar_key = FuncSymbol("bar")
    foo_key = FuncSymbol("foo")
    goo_key = FuncSymbol("goo")

    table.insert(foo)
    assert table.lookup(foo_key)

    table.insert(bar)
    assert table.lookup(bar_key)

    with pytest.raises(AssertionError):
        table.insert(foo)

    # test with scope
    with table.scope_guard():
        table.insert(goo)
        assert table.lookup(goo_key)

        assert table.lookup(foo_key)
        assert table.lookup(bar_key)

    # goo is removed after local scope
    assert not table.lookup(goo_key)


def test_func_table_override():
    table = FuncTable()

    arg0 = ast.ArgDecl(name="x", type=_ty.Int, loc=None)
    arg1 = ast.ArgDecl(name="y", type=_ty.Int, loc=None)
    arg2 = ast.ArgDecl(name="z", type=_ty.Int, loc=None)

    foo0 = FuncDecl(name="foo", args=[], return_type=None, loc=None, body=[])
    foo1 = FuncDecl(name="foo", args=[arg0], return_type=None, loc=None, body=[])
    foo2 = FuncDecl(name="foo", args=[arg0, arg1], return_type=None, loc=None, body=[])

    table.insert(foo0)
    table.insert(foo1)
    table.insert(foo2)

    funcs = table.lookup(FuncSymbol("foo"))
    assert len(funcs) == 3
