import pimacs.ast.ast_visitor as ast_visitor
from pimacs.ast.ast import UAttr, VarDecl, VarRef

from .ast import AnalyzedClass, LispCall, MakeObject, UCallMethod


class IRVisitor(ast_visitor.IRVisitor):
    def visit_AnalyzedClass(self, node):
        self.visit_Class(node)

    def visit_MakeObject(self, node):
        pass

    def visit_UCallMethod(self, node: UCallMethod):
        pass

    def visit_LispCall(self, node: LispCall):
        pass


class IRMutator(ast_visitor.IRMutator):
    def visit_AnalyzedClass(self, node):
        return self.visit_Class(node)

    def visit_MakeObject(self, node):
        return node

    def visit_UCallMethod(self, node):
        return node

    def visit_LispCall(self, node):
        return node


class IRPrinter(ast_visitor.IRPrinter):
    def visit_AnalyzedClass(self, node: AnalyzedClass):
        self.visit_Class(node)

    def visit_MakeObject(self, node):
        self.put(f"make_{node.class_name}()")

    def visit_UCallMethod(self, node):
        self.put("U<")
        self.visit(node.obj)
        self.put(".")
        self.put(f"{node.attr}")
        self.put(">")
        self.put("(")
        for i, arg in enumerate(node.args):
            if i > 0:
                self.put(", ")
            self.visit(arg)
        self.put(")")

    def visit_CallMethod(self, node):
        if isinstance(node.obj, VarDecl):
            self.put(node.obj.name)
        else:
            raise NotImplementedError(f"Unknown obj type: {type(node.obj)}")

        self.put(".")
        self.put(node.method.name)
        self.put("(")

        for i, arg in enumerate(node.args):
            if i > 0:
                self.put(", ")
            self.visit(arg)
        self.put(")")

    def visit_LispCall(self, node):
        self.put("%(")
        self.put(node.func)
        for no, arg in enumerate(node.args):
            if no > 0:
                self.put(" ")
            self.visit(arg)
        self.put(")")
