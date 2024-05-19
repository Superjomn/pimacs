import pimacs.ast.ast as ast
from pimacs.ast.type import Int, Str
from pimacs.sema.context import *


def test_type_singleton():
    ctx = ModuleContext("test")
    type_system = TypeSystem(ctx)

    list_int0 = type_system.get_ListType(Int)
    list_int1 = type_system.get_ListType(Int)

    assert list_int0 is list_int1

    # custom
    T0 = type_system.get_type_placeholder("T0")
    assert T0

    # Dict
    dict0 = type_system.get_DictType(Str, Int)
    dict1 = type_system.get_DictType(Str, Int)
    assert dict0 is dict1


def test_type_template():
    ctx = ModuleContext("test")
    type_system = TypeSystem(ctx)

    # Dict[T0, T2], not concrete
    T0 = type_system.get_type_placeholder("T0")
    T1 = type_system.get_type_placeholder("T1")
    dict_ty = type_system.get_DictType(T0, T1)
    assert not dict_ty.is_concrete

    # Dict[Int, T1], not concrete
    dict_ty = type_system.get_DictType(Int, T1)
    assert not dict_ty.is_concrete

    # Dict[Int, Str], concrete
    dict_ty = type_system.get_DictType(Int, Str)
    assert dict_ty.is_concrete


def test_type_alias():
    ctx = ModuleContext("test")
    symtbl = SymbolTable()
    type_system = TypeSystem(symtbl)

    with symtbl.scope_guard():
        T0 = type_system.get_type_placeholder("T0")
        type_system.add_type_alias("T0", T0)
        T0 = type_system.get_type("T0")
        assert T0
        T0_ = type_system.get_type("T0")
        assert T0 is T0_

    T0 = type_system.get_type("T0")
    assert not T0


def test_SymbolTable_var():
    table = SymbolTable()
    var0 = ast.VarDecl("var0", Int)
    var1 = ast.VarDecl("var1", Int)
    sym0 = Symbol("var0", kind=Symbol.Kind.Var)
    sym1 = Symbol("var1", kind=Symbol.Kind.Var)

    table.insert(sym0, var0)
    assert table.get_symbol(sym0)

    with table.scope_guard():
        table.insert(sym1, var1)
        assert table.get_symbol(sym1)
    assert not table.get_symbol(sym1)
    assert table.get_symbol(sym0)


def test_SymbolTable_func():
    table = SymbolTable()

    func = ast.Function(name="func", args=[],
                        return_type=Int, loc=None, body=[])
    func_sym = FuncSymbol("func")
    table.insert(func_sym, func)
    assert table.get_function(func_sym)

    arg0 = ast.Arg("x", Int)
    arg1 = ast.Arg("y", Int)
    func1 = ast.Function(name="func", args=[
                         arg0], return_type=Int, loc=None, body=[])
    func2 = ast.Function(
        name="func", args=[arg0, arg1], return_type=Int, loc=None, body=[]
    )

    with table.scope_guard():  # local functions
        # There is no function in the local scope
        assert not table.contains_locally(func_sym)

        table.insert(func_sym, func1)
        table.insert(func_sym, func2)

        funcs = table.get_function(func_sym)
        assert len(funcs) == 3

        assert table.contains_locally(func_sym)

    funcs = table.get_function(func_sym)
    assert len(funcs) == 1

    table.contains_locally(func_sym)
