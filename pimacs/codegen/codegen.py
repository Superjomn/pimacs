from typing import List as _List
from typing import Tuple

from pimacs.ast.ast_printer import PrinterBase
from pimacs.lisp.ast import *


class Codegen(PrinterBase):
    module_statment_newlines = 2

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
        raise NotImplementedError(
            f"No visit_{node.__class__.__name__} method for node {node}")

    def visit_NoneType(self, node):
        pass

    def visit_list(self, nodes: list, put_indent: bool = False, delimiter: str = " "):
        if not nodes:
            return

        for node in nodes[:-1]:
            if put_indent:
                self.put_indent()
            self.visit(node)
            self.put(delimiter)

        if put_indent:
            self.put_indent()
        self.visit(nodes[-1])

    def visit_str(self, node: str):
        self.put(node)

    def visit_stmts(self, nodes: _List[Node] | Tuple[Node], newlines: int = 1):
        if not nodes:
            return
        for node in nodes[:-1]:
            self.put_indent()
            self.visit(node)
            self.put("\n" * newlines)

        self.put_indent()
        self.visit(nodes[-1])

    def visit_Module(self, node: Module):
        self.visit_stmts(node.stmts, newlines=self.module_statment_newlines)

    def visit_Literal(self, node):
        self.put(str(node.value))

    def visit_VarDecl(self, node):
        self.put(node.name)

    def visit_VarRef(self, node: ast.VarRef):
        name = node.name or node.target.name  # type: ignore
        self.put(name)

    def visit_Symbol(self, node):
        self.put(node.name)

    def visit_List(self, node):
        self.put("(")
        self.visit_list(node.elements)
        self.put(")")

    def visit_Let(self, node: Let):
        self.put("(let (")
        self.visit_list(node.vars)
        self.put(")")
        self.put("\n")
        with self.indent_guard():
            self.visit_stmts(node.body)  # type: ignore
        self.put(")")

    def visit_Guard(self, node: Guard):
        self.put("(")
        self.visit(node.header)
        self.visit_list(node.body)
        self.put(")")

    def visit_Return(self, node: Return):
        self.put(f"(cl-return-from {node.block_name} ")
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

    def visit_Block(self, node: Block):
        self.put(f"(cl-block {node.name}\n")
        with self.indent_guard():
            self.visit_list(node.stmts, put_indent=True, delimiter="\n")
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

    def visit_If(self, node: If) -> None:
        """Visit an If node and generate code"""
        self.put("(if ")
        self.visit(node.cond)
        self.put("\n")

        with self.indent_guard():
            self.put_indent()
            self.visit_list(node.then_block, put_indent=True, delimiter="\n")
            if node.else_block:
                self.put_indent()
                self.visit_list(node.else_block,
                                put_indent=True, delimiter="\n")

        self.put(")")

    def visit_While(self, node: While):
        self.put("(while ")
        self.visit(node.cond)
        self.put(" ")
        self.visit(node.body)
        self.put(")")

    def visit_Struct(self, node: Struct):
        self.put(f"(cl-defstruct {node.name} ")
        self.visit(node.fields)
        self.put(")")
        if node.methods:
            self.put("\n" * self.module_statment_newlines)
            # type: ignore
            self.visit_stmts(
                node.methods, newlines=self.module_statment_newlines)  # type: ignore
