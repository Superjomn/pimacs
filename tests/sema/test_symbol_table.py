import pytest

from pimacs.ast import ast
from pimacs.sema.symbol_table import *


def test_Scope_var():
    scope = Scope()
    a = Symbol("a", Symbol.Kind.Var)
    var = ast.VarDecl(name="a", loc=None)

    scope.add(a, var)
    assert len(scope) == 1

    scope1 = Scope(parent=scope)
    var1 = ast.VarDecl(name="b", loc=None)
    b = Symbol("b", Symbol.Kind.Var)
    scope1.add(b, var1)
    assert len(scope1) == 1

    assert scope1.get(a) is var
    assert scope1.get(b) is var1

    assert scope.get(a) is var
    assert scope.get(b) is None


def test_func_table():
    table = SymbolTable()

    bar_key = FuncSymbol("bar")
    foo_key = FuncSymbol("foo")
    goo_key = FuncSymbol("goo")

    assert not table.get_function(foo_key)
    assert not table.get_function(bar_key)

    foo = ast.Function(name="foo", args=[],
                       return_type=None, loc=None, body=[])
    bar = ast.Function(name="bar", args=[],
                       return_type=None, loc=None, body=[])
    goo = ast.Function(name="goo", args=[],
                       return_type=None, loc=None, body=[])

    table.insert(foo)
    assert table.get_symbol(foo_key)

    table.insert(bar)
    assert table.get_symbol(bar_key)

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

    arg0 = ast.Arg(name="x", type=_ty.Int, loc=None)
    arg1 = ast.Arg(name="y", type=_ty.Int, loc=None)
    arg2 = ast.Arg(name="z", type=_ty.Int, loc=None)

    foo0 = Function(name="foo", args=[], return_type=None, loc=None, body=[])
    foo1 = Function(name="foo", args=[arg0],
                    return_type=None, loc=None, body=[])
    foo2 = Function(name="foo", args=[arg0, arg1],
                    return_type=None, loc=None, body=[])

    table.insert(foo0)
    table.insert(foo1)
    table.insert(foo2)

    funcs = table.lookup(FuncSymbol("foo"))
    assert len(funcs) == 3

    unresolved_func = ast.UFunction(name="foo", loc=None)

    param0 = ast.CallParam(
        name="x", value=ast.make_const(0, loc=None), loc=None)
    param1 = ast.CallParam(
        name="y", value=ast.make_const(0, loc=None), loc=None)
    param2 = ast.CallParam(
        name="z", value=ast.make_const(0, loc=None), loc=None)

    call = ast.Call(func=unresolved_func, args=tuple(), loc=None)
    target_func = funcs.lookup(call.args)
    assert target_func is foo0

    call = ast.Call(func=unresolved_func, args=(param0,), loc=None)
    target_func = funcs.lookup(call.args)
    assert target_func is foo1

    call = ast.Call(func=unresolved_func, args=(param0, param1), loc=None)
    target_func = funcs.lookup(call.args)
    assert target_func is foo2
