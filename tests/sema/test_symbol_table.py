from pimacs.ast import ast
from pimacs.sema.symbol_table import *


def test_Scope_non_func():
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
