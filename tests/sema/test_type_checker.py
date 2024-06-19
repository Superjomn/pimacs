from pprint import pprint

import pytest

import pimacs.ast.type as _ty
from pimacs.sema import SemaError
from pimacs.sema.type_checker import *
from pimacs.transpiler.phases import *


def test_VarDecl():
    ctx = ModuleContext()
    ti = TypeChecker(ctx)
    node = ast.VarDecl(name='a', init=ast.Literal(
        value=1, loc=None), loc=None)
    ti(node)
    assert node.type == _ty.Int


def test_VarDecl_chain():
    ctx = ModuleContext()

    code = '''
var a = 1
var b = a
var c = b
    '''

    tree = parse_ast(code.rstrip())
    tree = perform_sema(ctx, tree)

    type_inferene = TypeChecker(ctx)
    type_inferene(tree)
    pprint(tree)

    assert len(tree.stmts) == 3
    for s in tree.stmts:
        assert s.type == _ty.Int


def test_call():
    ctx = ModuleContext()
    code = '''
var a = foo()
var b = a
var c = a + b

def foo() -> Int:
    return 1
    '''
    tree = parse_ast(code.rstrip())
    tree = perform_sema(ctx, tree)
    pprint(tree)

    type_inferene = TypeChecker(ctx)
    type_inferene(tree)
    pprint(tree)


def test_call_conflict():
    ctx = ModuleContext(enable_exception=True)
    code = '''
var a: Float = foo()
def foo() -> Int:
    return 1
'''
    tree = parse_ast(code.rstrip())

    with pytest.raises(SemaError):
        tree = perform_sema(ctx, tree)

        pprint(tree)


def test_amend_placeholder_types():
    ctx = ModuleContext()
    code = '''
@template[T0, T1]
def foo(a: T0, b: T1) -> T0:
    var c: T0 = a + b
    return c
'''
    tree = parse_ast(code.rstrip())

    func: ast.Function = tree.stmts[0]
    pprint(func)

    mapping = {
        _ty.GenericType('T0'): _ty.PlaceholderType('PT0'),
        _ty.GenericType('T1'): _ty.PlaceholderType('PT1'),
    }

    amend_placeholder_types(tree, mapping)
    print("after")
    pprint(func)

    for arg in func.args:
        assert isinstance(arg.type, _ty.PlaceholderType)
    assert isinstance(func.return_type, _ty.PlaceholderType)
    for stmt in func.body.stmts:
        if isinstance(stmt, ast.VarDecl):
            assert isinstance(stmt.type, _ty.PlaceholderType)


def test_amend_placeholder_types_class():
    ctx = ModuleContext()
    code = '''
@template[T]
class App:
    @template[T0, T1]
    def foo(a: T0, b: T1) -> T:
        return a + b
'''
    tree = parse_ast(code.rstrip())

    cls: ast.Class = tree.stmts[0]
    pprint(cls)

    mapping = {
        _ty.GenericType('T'): _ty.PlaceholderType('PT'),
        _ty.GenericType('T0'): _ty.PlaceholderType('PT0'),
        _ty.GenericType('T1'): _ty.PlaceholderType('PT1'),
    }

    amend_placeholder_types(tree, mapping)
    print("after")
    pprint(cls)

    for method in cls.body:
        for arg in method.args:
            assert isinstance(arg.type, _ty.PlaceholderType)
        assert isinstance(method.return_type, _ty.PlaceholderType)


if __name__ == '__main__':
    test_amend_placeholder_types_class()
