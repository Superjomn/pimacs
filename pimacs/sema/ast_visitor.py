from io import StringIO

import pimacs.ast.ast_visitor as ast_visitor
from pimacs.ast.ast import VarRef
from pimacs.ast.ast_printer import IRPrinter as _IRPrinter

from .ast import AnalyzedClass, UCallMethod


class IRVisitor(ast_visitor.IRVisitor):
    def visit_AnalyzedClass(self, node):
        self.visit_Class(node)

    def visit_MakeObject(self, node):
        pass

    def visit_UCallMethod(self, node: UCallMethod):
        pass

    def visit_CallMethod(self, node):
        pass


class IRMutator(ast_visitor.IRMutator):
    def visit_AnalyzedClass(self, node):
        return self.visit_Class(node)

    def visit_MakeObject(self, node):
        return node

    def visit_UCallMethod(self, node: UCallMethod):
        with node.write_guard():
            node.obj = self.visit(node.obj)
            node.args = self.visit(node.args)
        return node

    def visit_CallMethod(self, node):
        with node.write_guard():
            node.obj = self.visit(node.obj)
            node.args = self.visit(node.args)
        return node


class IRPrinter(_IRPrinter):

    def __init__(self, os, mark_unresolved=False) -> None:
        super().__init__(os, mark_unresolved=mark_unresolved)

    def visit_AnalyzedClass(self, node: AnalyzedClass):
        self.visit_Class(node)

    def visit_MakeObject(self, node):
        self.put(f"make_obj[{node.type}]()")

    def visit_UCallMethod(self, node):
        if self._mark_unresolved:
            self.put("U<")
        self.visit(node.obj)
        self.put(".")
        self.put(f"{node.attr}")
        if self._mark_unresolved:
            self.put(">")
        self.put("(")
        for i, arg in enumerate(node.args):
            if i > 0:
                self.put(", ")
            self.visit(arg)
        self.put(")")

    def visit_CallMethod(self, node):
        if isinstance(node.obj, VarRef):
            name = node.obj.name or node.obj.target.name
        else:
            raise Exception(f"Unexpected obj type {node.obj}")
        self.put(name)

        self.put(".")
        self.put(node.method.name)
        self.put("(")

        for i, arg in enumerate(node.args):
            if i > 0:
                self.put(", ")
            self.visit(arg)
        self.put(")")


def print_ast(node, mark_unresolved=True):
    printer = IRPrinter(StringIO(), mark_unresolved=mark_unresolved)
    printer(node)

    output = printer.os.getvalue()
    print(output)
