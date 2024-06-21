from io import StringIO
from pathlib import Path
from typing import Set

import pytest

import pimacs.ast.type as _ty
from pimacs.sema.ast_visitor import print_ast
from pimacs.sema.ast_walker import ASTWalker, Traversal
from pimacs.sema.context import ModuleContext
from pimacs.sema.file_sema import *
from pimacs.transpiler.phases import parse_ast, perform_sema


def find_unresolved_symbols(node) -> Set[ast.Node]:
    class Walker(ASTWalker):
        def __init__(self):
            self.unresolved_symbols = set()

        def walk_to_node_post(self, node):
            return node

        def walk_to_node_pre(self, node) -> bool:
            if isinstance(node, ast.Node) and not node.resolved:
                self.unresolved_symbols.add(node)
            return True

    walker = Walker()
    Traversal(walker)(node)

    return walker.unresolved_symbols


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
    assert isinstance(call.target, ast.Function)


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


def test_template_class():
    code = '''
@template[T0, T1]
class App:
    var a: T0
    var b: T1

    def __init__(self, a: T0, b: T1):
        self.a = a
        self.b = b

var app = App(1, 2)
var app1 = App(1, 2.0)
'''
    ctx = ModuleContext(enable_exception=True)
    file = parse_ast(code.rstrip())
    file = perform_sema(ctx, file)
    pprint(file)

    unresolved = find_unresolved_symbols(file)
    assert not unresolved

    class_def = file.stmts[0]
    assert isinstance(class_def, AnalyzedClass)

    print("code:")
    print_ast(file)


def test_load_builtins():
    ''' Test the builtin modules, and keep them pass the Sema.'''
    builtin_root = Path(os.path.join(
        os.path.dirname(__file__), "../../pimacs/builtin"))

    def load(path):
        ctx = ModuleContext(enable_exception=True)
        file = parse_ast(filename=path)
        file = perform_sema(ctx, file)
        return file

    file = load(builtin_root / "list.pis")
    unresolved = find_unresolved_symbols(file)
    assert not unresolved


if __name__ == '__main__':
    # test_UVarRefScope()
    # test_func_binding()
    # test_func_binding()
    # test_class()
    test_template_class()
