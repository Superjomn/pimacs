from io import StringIO

import pytest

from pimacs.sema.ast_visitor import IRPrinter, StringStream
from pimacs.sema.context import ModuleContext
from pimacs.sema.file_sema import *
from pimacs.transpiler.phases import parse_ast, perform_sema


def test_UVarRef():
    code = '''
var a = 1
a = 2'''
    file = parse_ast(code)
    assign_stmt = file.stmts[1]
    assert isinstance(assign_stmt, ast.Assign)
    assert isinstance(assign_stmt.target, ast.UVarRef)  # unresolved symbol

    ctx = ModuleContext()
    file = perform_sema(ctx, file)
    print(file)
    assign_stmt = file.stmts[1]
    assert isinstance(assign_stmt, ast.Assign)
    assert isinstance(assign_stmt.target, ast.VarRef)  # resolved symbol


def test_UVarRefScope():
    code = '''
var a = 1
def foo():
    a = 2'''

    file = parse_ast(code)
    file = perform_sema(ModuleContext(), file)
    print(file)
    fn = file.stmts[1]
    assert isinstance(fn, ast.Function)
    assign_stmt = fn.body.stmts[0]
    assert isinstance(assign_stmt, ast.Assign)
    assert isinstance(assign_stmt.target, ast.VarRef)  # resolved symbol


def test_func_binding():
    code = '''
def foo():
    return

foo()'''
    file = parse_ast(code)
    file = perform_sema(ModuleContext(), file)
    print(file)

    call = file.stmts[1]
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Function)


def test_class():
    code = '''
class App:
    var a = 1
    var b = 2
'''
    file = parse_ast(code.rstrip())
    file = perform_sema(ModuleContext(), file)
    print(file)
    class_def = file.stmts[0]
    assert isinstance(class_def, AnalyzedClass)

    printer = IRPrinter(StringIO())
    printer(class_def)
    print("printer", printer.os.getvalue())
    assert class_def.name == "App"
    assert len(class_def.symbols) == 2  # a, b declared in class


if __name__ == '__main__':
    # test_UVarRefScope()
    # test_func_binding()
    # test_func_binding()
    test_class()
