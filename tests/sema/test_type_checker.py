from pprint import pprint

import pytest

import pimacs.ast.type as _ty
from pimacs.codegen.phases import *
from pimacs.sema.context import SemaError
from pimacs.sema.type_checker import *  # type: ignore


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
var a: Bool = foo()
def foo() -> Int:
    return 1
'''
    tree = parse_ast(code.rstrip())

    with pytest.raises(SemaError):
        tree = perform_sema(ctx, tree)

        pprint(tree)


def test_amend_placeholder_types():
    code = '''
def foo[T0, T1](a: T0, b: T1) -> T0:
    var c: T0 = a + b
    return c
'''
    tree = parse_ast(code.rstrip())

    func: ast.Function = tree.stmts[0]
    pprint(func)

    for arg in func.args:
        assert isinstance(arg.type, _ty.PlaceholderType), type(arg.type)
    assert isinstance(func.return_type, _ty.PlaceholderType)
    for stmt in func.body.stmts:
        if isinstance(stmt, ast.VarDecl):
            assert isinstance(stmt.type, _ty.PlaceholderType)


def test_amend_placeholder_types_class():
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

    for method in cls.body:
        for arg in method.args:
            assert isinstance(arg.type, _ty.PlaceholderType)
        assert isinstance(method.return_type, _ty.PlaceholderType)


def test_convert_to_template_param_type():
    ctx = ModuleContext()
    code = '''
def foo[T](a: T) -> T:
    var b = a + 1
    return a + b
'''
    tree = parse_ast(code.rstrip())

    pprint(tree)

    tree = perform_sema(ctx, tree)

    func: ast.Function = tree.stmts[0]
    pprint(func)

    assert len(func.tp_assumptions) == 1


if __name__ == '__main__':
    test_amend_placeholder_types_class()
    test_convert_to_template_param_type()
    test_amend_placeholder_types()
