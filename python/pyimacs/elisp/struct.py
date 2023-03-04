import ast
import inspect
from typing import *

import astpretty


class ClassCompiler(ast.NodeVisitor):
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        node.body = [self.visit(b) for b in node.body]


def struct(class__):
    source = inspect.getsource(class__)
    class_ast = ast.parse(source)
    astpretty.pprint(class_ast)
    return class__
