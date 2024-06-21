from contextlib import contextmanager

import pimacs.ast.type as ty

from . import ast
from .ast_visitor import IRVisitor


class IRPrinter(IRVisitor):
    indent_width = 4

    def __init__(self, os, mark_unresolved=False) -> None:
        self.os = os
        self._indent: int = 0
        self._mark_unresolved = mark_unresolved

    def __call__(self, node: ast.Node) -> None:
        self.visit(node)

    def put_indent(self) -> None:
        self.os.write(" " * self._indent * self.indent_width)

    def put(self, s: str) -> None:
        self.os.write(s)

    def indent(self) -> None:
        self._indent += 1

    def deindent(self) -> None:
        self._indent -= 1

    def visit_VarDecl(self, node: ast.VarDecl):
        for no, decorator in enumerate(node.decorators):
            if no >= 1:
                self.put_indent()
            self.visit(decorator)
            self.put("\n")
        if node.mutable:
            self.put(f"var {node.name}")
        else:
            self.put(f"let {node.name}")
        if node.type is not None:
            self.put(" :")
            self.put(str(node.type))
        if node.init is not None:
            self.put(" = ")
            self.visit(node.init)

    def visit_Literal(self, node: ast.Literal):
        if node.value is not None:
            self.put(str(node.value))
        else:
            self.put("nil")

    def visit_Type(self, node: ast.Type):
        self.put(str(node))

    def visit_Function(self, node: ast.Function):
        for no, decorator in enumerate(node.decorators):
            if no >= 1:
                self.put_indent()
            self.visit(decorator)
            self.put("\n")

        if node.decorators:
            self.put_indent()
        self.put(f"def {node.name} (")
        if node.args:
            for i, arg in enumerate(node.args):
                if i > 0:
                    self.put(", ")
                self.visit(arg)
        self.put(")")

        if node.return_type is not None:
            self.put(" -> ")
            self.visit(node.return_type)

        self.put(":\n")

        self.visit(node.body)

    def visit_Arg(self, node: ast.Arg):
        self.put(f"{node.name}")
        if node.type is not None:
            self.put(" :")
            self.visit(node.type)
        if node.default is not None:
            self.put(" = ")
            self.visit(node.default)

    def visit_Call(self, node: ast.Call):
        match type(node.target):
            case ast.VarRef:
                self.put(f"{node.target.name}")  # type: ignore
            case ast.Function:
                self.put(f"{node.target.name}")  # type: ignore
            case ast.Class:
                self.put(f"{node.target.name}")
            case ast.Arg:
                self.put(f"{node.target.name}")  # type: ignore
            case ast.VarDecl:
                assert node.target.init is not None
                self.visit(node.target.init)
            case ast.UVarRef:
                if self._mark_unresolved:
                    self.put(f"UV<{node.target.name}>")  # type: ignore
                else:
                    self.put(f"{node.target.name}")  # type: ignore
            case ast.UFunction:
                if self._mark_unresolved:
                    self.put(f"UF<{node.target.name}>")  # type: ignore
                else:
                    self.put(f"{node.target.name}")  # type: ignore
            case ast.UAttr:
                self.visit(node.target)
            case _:
                if isinstance(node.target, str):
                    self.put(f"{node.target}")
                else:
                    raise Exception(f"{node.loc}\nInvalid function call: {
                                    node.target}, type: {type(node.target)}")

        if node.type_spec:
            self.put("[")
            for i, t in enumerate(node.type_spec):
                if i > 0:
                    self.put(", ")
                self.visit(t)
            self.put("]")

        self.put("(")
        for i, arg in enumerate(node.args):
            if i > 0:
                self.put(", ")
            self.visit(arg)
        self.put(")")

    def visit_Block(self, node: ast.Block):
        with self.indent_guard():
            if node.doc_string is not None:
                self.put_indent()
                self.visit(node.doc_string)
                self.put("\n")
            for stmt in node.stmts:
                self.put_indent()
                self.visit(stmt)
                self.put("\n")

    def visit_BinaryOp(self, node: ast.BinaryOp):
        self.visit(node.left)
        self.put(f" {node.op.value} ")
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.put("(")
        self.put(f"{node.op.value}")
        self.visit(node.operand)
        self.put(")")

    def visit_VarRef(self, node: ast.VarRef):
        if node.name:
            self.put(node.name)
        elif node.target is not None:
            self.put(node.target.name)  # type: ignore

    def visit_If(self, node: ast.If):
        self.put("if ")
        self.visit(node.cond)
        self.put(":\n")
        self.visit(node.then_branch)
        for cond, block in node.elif_branches:
            self.put_indent()
            self.put("elif ")
            self.visit(cond)
            self.put(":\n")
            self.visit(block)
        if node.else_branch is not None:
            self.put_indent()
            self.put("else:\n")
            self.visit(node.else_branch)

    def visit_Return(self, node: ast.Return):
        self.put("return")
        if node.value is not None:
            self.put(" ")
            self.visit(node.value)

    def visit_File(self, node: ast.File):
        for stmt in node.stmts:
            if isinstance(stmt, ast.Function):
                self.put("\n")
            self.put_indent()
            self.visit(stmt)
            self.put("\n")

    def visit_Decorator(self, node: ast.Decorator):
        if isinstance(node.action, ast.Call):
            self.put(f"@{node.action.target.name}(")  # type: ignore
            if node.action.args:
                for i, arg in enumerate(node.action.args):
                    if i > 0:
                        self.put(", ")
                    self.visit(arg)
            self.put(")")
        elif isinstance(node.action, str):
            self.put(f"@{node.action}")
        elif isinstance(node.action, ast.Template):
            self.put(f"@template[")
            for i, t in enumerate(node.action.types):
                if i > 0:
                    self.put(", ")
                self.visit(t)
            self.put("]")
        else:
            raise Exception(f"Invalid decorator action: {node.action}")

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.target)
        self.put(" = ")
        self.visit(node.value)

    def visit_Class(self, node: ast.Class):
        for no, decorator in enumerate(node.decorators):
            if no >= 1:
                self.put_indent()
            self.visit(decorator)
            self.put("\n")
        self.put(f"class {node.name}:\n")
        with self.indent_guard():
            for stmt in node.body:
                if isinstance(stmt, ast.Function):
                    self.put("\n")
                self.put_indent()
                self.visit(stmt)
                self.put("\n")

    def visit_UVarRef(self, node: ast.UVarRef):
        self.put(f"{node.name}")

    def visit_UAttr(self, node: ast.UAttr):
        self.visit(node.value)
        self.put(".")
        if self._mark_unresolved:
            self.put("U<")
        self.put(node.attr)
        if self._mark_unresolved:
            self.put(">")

    def visit_DocString(self, node: ast.DocString):
        self.put(f'"{node.content}"')

    def visit_Select(self, node: ast.Select):
        self.visit(node.then_expr)
        self.put(" if ")
        self.visit(node.cond)
        self.put(" else ")
        self.visit(node.else_expr)

    def visit_Guard(self, node: ast.Guard):
        self.put("guard ")
        self.visit(node.header)
        self.put(":\n")
        self.visit(node.body)

    def visit_CallParam(self, node: ast.CallParam):
        if node.name:
            self.put(f"{node.name} = ")
        self.visit(node.value)

    def visit_GenericType(self, node: ty.GenericType):
        self.put(str(node))

    def visit_CompositeType(self, node: ty.CompositeType):
        self.put(str(node))

    def visit_IntType(self, node: ty.IntType):
        self.put(str(node))

    def visit_FloatType(self, node: ty.FloatType):
        self.put(str(node))

    def visit_BoolType(self, node: ty.BoolType):
        self.put(str(node))

    def visit_StrType(self, node: ty.StrType):
        self.put(str(node))

    def visit_UnkType(self, node: ty.UnkType):
        self.put(str(node))

    def visit_NilType(self, node: ty.NilType):
        self.put(str(node))

    def visit_LispType_(self, node: ty.LispType_):
        self.put(str(node))

    def visit_PlaceholderType(self, node: ty.PlaceholderType):
        self.put(str(node))

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.VarRef):
            self.put(node.value.target.name)  # type: ignore
        else:
            self.put(node.value.name)
        self.put(f".{node.attr}")

    def visit_UFunction(self, node: ast.UFunction):
        if self._mark_unresolved:
            self.put(f"UFunction<{node.name}>")
        else:
            self.put(f"{node.name}")

    def visit_UClass(self, node: ast.UClass):
        if self._mark_unresolved:
            self.put(f"UClass<{node.name}>")
        else:
            self.put(f"{node.name}")

    def visit_MakeObject(self, node):
        self.put(f"{type}()")

    def visit_LispCall(self, node: ast.LispCall):
        self.put("%(")
        self.put(node.target)
        for no, arg in enumerate(node.args):
            self.put(" ")
            self.visit(arg)
        self.put(")")

    @contextmanager
    def indent_guard(self):
        self.indent()
        yield
        self.deindent()
