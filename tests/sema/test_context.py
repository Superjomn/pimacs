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


# TODO: Enable this test
def _test_type_alias():
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
