from pprint import pprint

from pimacs.codegen.phases import parse_ast, perform_sema, translate_to_lisp
from pimacs.lisp.ast import *
from pimacs.sema.context import ModuleContext


def get_lisp(code: str) -> Module:
    the_ast = parse_ast(code.strip())
    ctx = ModuleContext(enable_exception=True)
    the_ast = perform_sema(ctx, the_ast)  # type: ignore
    assert the_ast
    print('ast after Sema:')
    pprint(the_ast)
    module = translate_to_lisp(ctx, the_ast)
    return module


def test_VarDecl():
    code = '''
var a: Int = 1
    '''
    module = get_lisp(code)

    assign = module.stmts[0]
    assert isinstance(assign, Assign)


def test_Function():
    code = '''
def min(a:Int, b:Int) -> Int:
    return a if a < b else b
    '''

    module = get_lisp(code)
    pprint(module)


if __name__ == "__main__":
    test_VarDecl()
    test_Function()
