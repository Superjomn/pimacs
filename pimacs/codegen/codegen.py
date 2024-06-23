from typing import List as _List
from typing import Tuple

from pimacs.ast.ast_printer import PrinterBase
from pimacs.lisp.ast import *


class Codegen(PrinterBase):
    def __init__(self, os):
        super().__init__(os, indent_width=2)

    def __call__(self, node):
        self.visit(node)
        return self.os.getvalue()

    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)

    def generic_visit(self, node):
        raise NotImplementedError(f"No visit_{node.__class__.__name__} method")

    def visit_list(self, nodes):
        if not nodes:
            return

        for node in nodes[:-1]:
            self.visit(node)
            self.put(" ")
        self.visit(nodes[-1])

    def visit_str(self, node: str):
        self.put(node)

    def visit_stmts(self, nodes: _List[Node] | Tuple[Node]):
        if not nodes:
            return
        for node in nodes[:-1]:
            self.put_indent()
            self.visit(node)
            self.put("\n\n")

        self.put_indent()
        self.visit(nodes[-1])

    def visit_Module(self, node: Module):
        self.visit_stmts(node.stmts)

    def visit_Literal(self, node):
        self.put(str(node.value))

    def visit_VarDecl(self, node):
        # The VarDecl should be handled in the Let node for local scope or the Assign node in global scope
        assert NotImplementedError("No VarDecl is allowed")

    def visit_VarRef(self, node: ast.VarRef):
        name = node.name or node.target.name  # type: ignore
        self.put(name)

    def visit_Symbol(self, node):
        self.put(node.name)

    def visit_List(self, node):
        self.put("(")
        self.visit_list(node.elements)
        self.put(")")

    def visit_Let(self, node):
        self.put("(let (")
        self.visit_list(node.vars)
        self.put(")")
        self.put("\n")
        with self.indent_guard():
            self.visit_stmts(node.body)
        self.put(")")

    def visit_Guard(self, node: Guard):
        self.put("(")
        self.visit(node.header)
        self.visit_list(node.body)
        self.put(")")

    def visit_Return(self, node):
        self.put("(cl-return ")
        self.visit(node.value)
        self.put(")")

    def visit_Assign(self, node: Assign):
        self.put("(")
        if isinstance(node.target, Attribute):
            self.put("setf ")
        else:
            self.put("setq ")
        self.visit(node.target)
        self.put(" ")
        self.visit(node.value)
        self.put(")")

    def visit_Attribute(self, node: Attribute):
        self.put("(")
        self.put(f"{node.class_name}-{node.attr} ")
        self.put(f"{node.target.name}")
        self.put(")")

    def visit_Function(self, node: Function):
        self.put("(defun ")

        self.put(node.name)
        self.put(" (")
        self.visit(node.args)
        self.put(")")
        self.put("\n")

        with self.indent_guard():
            self.put_indent()
            self.visit(node.body)
        self.put(")")

    def visit_Call(self, node: Call):
        self.put("(")
        self.visit(node.func)
        self.visit_list(node.args)
        self.put(")")

    def visit_If(self, node: If):
        self.put("(if ")
        self.visit(node.cond)
        self.put(" ")

        self.visit(node.then_block)
        self.put(" ")
        self.visit(node.else_block)
        self.put(")")

    def visit_While(self, node: While):
        self.put("(while ")
        self.visit(node.cond)
        self.put(" ")
        self.visit(node.body)
        self.put(")")

    def visit_Struct(self, node: Struct):
        self.put("(cl-defstruct ")
        self.put(node.name)
        self.put(" ")
        self.visit(node.fields)
        self.put(")")
