import pimacs.ast.ast_visitor as ast_visitor

from .ast import AnalyzedClass, MakeObject


class IRVisitor(ast_visitor.IRVisitor):
    def visit_AnalyzedClass(self, node):
        self.visit_Class(node)

    def visit_MakeObject(self, node):
        pass


class IRMutator(ast_visitor.IRMutator):
    def visit_AnalyzedClass(self, node):
        return self.visit_Class(node)

    def visit_MakeObject(self, node):
        return node


class IRPrinter(ast_visitor.IRPrinter):
    def visit_AnalyzedClass(self, node: AnalyzedClass):
        self.visit_Class(node)

    def visit_MakeObject(self, node):
        self.put(f"make_{node.class_name}()")
