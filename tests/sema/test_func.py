from pprint import pprint

import pytest

import pimacs.ast.ast as ast
import pimacs.ast.type as _ty
from pimacs.sema.context import ModuleContext
from pimacs.sema.func import *
from pimacs.sema.utils import *
from pimacs.transpiler.phases import *


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
    func = Function(name="foo", args=tuple(),
                    return_type=None, loc=None, body=[])
    sig = FuncSig.create(func)
    assert sig.symbol.name == "foo"
    assert not sig.input_types
    assert sig.output_type == _ty.Nil

    # foo(x: Int) -> nil
    func = Function(
        name="foo",
        args=(ast.Arg(name="x", type=_ty.Int, loc=None),),
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
        args=(ast.Arg(name="x", type=_ty.Int, loc=None),),
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
        args=(
            ast.Arg(name="x", type=_ty.Int, loc=None),
            ast.Arg(name="y", type=_ty.Int, loc=None),
        ),
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


def test_func_sig_template():
    code = '''
@template[T0, T1]
def foo(x: T0, y: T1) -> T0:
    return x

var a = foo(1, 2)
'''
    ctx = ModuleContext(enable_exception=True)
    tree = parse_ast(code)
    tree = perform_sema(ctx, tree)
    pprint(tree)


def test_func_sig_specialize():
    T0 = _ty.PlaceholderType(name="T0")
    T1 = _ty.PlaceholderType(name="T1")
    func = Function(
        name="foo",
        args=(
            ast.Arg(name="x", type=T0, loc=None),
            ast.Arg(name="y", type=T1, loc=None),
        ),
        return_type=T1,
        loc=None,
        body=[],
    )
    sig = FuncSig.create(func)

    mapping = {
        T0: _ty.Int,
        T1: _ty.Float,
    }

    new_sig = sig.specialize(mapping)
    assert new_sig.all_param_types_concrete()


def test_func_sig_specialize_List_T():
    T = _ty.PlaceholderType(name="T")
    List_T = _ty.CompositeType(name="List", params=(T,))
    func = Function(name="List",
                    return_type=List_T,
                    loc=None,
                    body=[],
                    )

    sig = FuncSig.create(func)
    assert not sig.all_param_types_concrete()

    mapping = {
        T: _ty.Int,
    }

    new_sig = sig.specialize(mapping)
    assert new_sig.all_param_types_concrete()


def test_func_sig_specialize_List_T1():
    T = _ty.PlaceholderType(name="T")
    List_T = _ty.CompositeType(name="List", params=(T,))
    func = Function(name="List",
                    return_type=List_T,
                    loc=None,
                    body=[],
                    )

    sig = FuncSig.create(func)
    assert not sig.all_param_types_concrete()

    mapping = {
        T: _ty.Int,
    }

    new_sig = sig.specialize(mapping)
    assert new_sig.all_param_types_concrete()


if __name__ == "__main__":
    test_func_sig_specialize()
