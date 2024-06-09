import pytest

import pimacs.ast.type as _ty
from pimacs.ast import ast
from pimacs.sema.func import FuncDuplicationError
from pimacs.sema.symbol_table import *
from pimacs.sema.utils import FuncSymbol, Symbol


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


def test_SymbolTable_func():
    table = SymbolTable()

    bar_key = FuncSymbol("bar")
    foo_key = FuncSymbol("foo")
    goo_key = FuncSymbol("goo")

    assert not table.lookup(foo_key)
    assert not table.lookup(bar_key)

    foo = ast.Function(name="foo", args=[],
                       return_type=None, loc=None, body=[])
    bar = ast.Function(name="bar", args=[],
                       return_type=None, loc=None, body=[])
    goo = ast.Function(name="goo", args=[],
                       return_type=None, loc=None, body=[])

    table.insert(foo_key, foo)
    assert table.lookup(foo_key)

    table.insert(bar_key, bar)
    assert table.lookup(bar_key)

    with pytest.raises(FuncDuplicationError):
        table.insert(foo_key, foo)

    # test with scope
    with table.scope_guard():
        table.insert(goo)
        assert table.lookup(goo_key)

        assert table.lookup(foo_key)
        assert table.lookup(bar_key)

    # goo is removed after local scope
    assert not table.lookup(goo_key)


def test_SymbolTable_func_override():
    table = SymbolTable()

    arg0 = ast.Arg(name="x", type=_ty.Int, loc=None)
    arg1 = ast.Arg(name="y", type=_ty.Int, loc=None)
    arg2 = ast.Arg(name="z", type=_ty.Int, loc=None)

    foo0 = ast.Function(name="foo", args=[],
                        return_type=None, loc=None, body=[])
    foo1 = ast.Function(name="foo", args=[arg0],
                        return_type=None, loc=None, body=[])
    foo2 = ast.Function(name="foo", args=[arg0, arg1],
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
    target_func_candidates = funcs.lookup(call.args)
    assert len(target_func_candidates) == 1
    assert target_func_candidates[0][0] is foo0

    call = ast.Call(func=unresolved_func, args=(param0,), loc=None)
    target_func_candidates = funcs.lookup(call.args)
    assert target_func_candidates[0][0] is foo1

    call = ast.Call(func=unresolved_func, args=(param0, param1), loc=None)
    target_func_candidates = funcs.lookup(call.args)
    assert target_func_candidates[0][0] is foo2


def test_SymbolTable_class_method():
    table = SymbolTable()

    self = ast.Arg(name="self", type=_ty.GenericType("App"),
                   loc=None, kind=ast.Arg.Kind.self_placeholder)

    foo0 = ast.Function(name="foo", args=[self],
                        return_type=None, loc=None, body=[])
    symbol = FuncSymbol("foo", annotation=ast.Function.Annotation.Class_method)

    table.insert(symbol, foo0)

    non_method_symbol = FuncSymbol("foo")
    assert not table.lookup(non_method_symbol)

    overloads: FuncOverloads = table.lookup(symbol)
    candidates = overloads.lookup(tuple())
    assert len(candidates) == 1


def test_SymbolTable_basic():
    a = Symbol("a", Symbol.Kind.Var)
    var = ast.VarDecl(name="a", loc=None)

    b = Symbol("b", Symbol.Kind.Var)
    var1 = ast.VarDecl(name="b", loc=None)

    st = SymbolTable()
    st.insert(a, var)

    assert st.lookup(a) is var

    with st.scope_guard():
        st.insert(b, var1)
        assert st.lookup(b) is var1
        assert st.lookup(a) is var

    assert st.lookup(b) is None


def test_SymbolTable_scope_hierarchy():
    st = SymbolTable()
    with st.scope_guard():
        with st.scope_guard():
            with st.scope_guard():
                assert len(st._scopes) == 4
                assert st._scopes[-1].parent == st._scopes[-2]
                assert st._scopes[-2].parent == st._scopes[-3]
