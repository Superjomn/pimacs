from pprint import pprint

import pytest

import pimacs.ast.type as _ty
from pimacs.sema import SemaError
from pimacs.sema.type_inference import *
from pimacs.transpiler.phases import *


def test_VarDecl():
    ctx = ModuleContext()
    ti = TypeInference(ctx)
    node = ast.VarDecl(name='a', init=ast.Constant(
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

    type_inferene = TypeInference(ctx)
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

    type_inferene = TypeInference(ctx)
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


if __name__ == '__main__':
    test_call_conflict()
