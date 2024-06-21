import logging
from contextlib import contextmanager

import pimacs.ast.ast as ast
import pimacs.ast.type as ty
from pimacs.logger import logger


class IRVisitor:
    def visit(self, node: ast.Node | ty.Type | str | None):
        if node is None:
            return
        if node is str:
            return node
        method_name = f"visit_{node.__class__.__name__}"
        logging.debug(f"Visiting {method_name}: {node}")
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.Node | ty.Type | str):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_FileName(self, node: ast.FileName):
        return node

    def visit_VarDecl(self, node: ast.VarDecl):
        self.visit(node.type)
        for decorator in node.decorators:
            self.visit(decorator)
        self.visit(node.init)

    def visit_Literal(self, node: ast.Literal):
        pass

    def visit_Template(self, node: ast.Template):
        for t in node.types:
            self.visit(t)

    def visit_Call(self, node: ast.Call):
        self.visit(node.target)
        for arg in node.args:
            self.visit(arg)

    def visit_CallParam(self, node: ast.CallParam):
        self.visit(node.value)

    def visit_int(self, node: int):
        pass

    def visit_float(self, node: float):
        pass

    def visit_Type(self, node: ast.Type):
        pass

    def visit_IntType(self, node: ty.IntType):
        pass

    def visit_FloatType(self, node: ty.FloatType):
        pass

    def visit_BoolType(self, node: ty.BoolType):
        pass

    def visit_StrType(self, node: ty.StrType):
        pass

    def visit_UnkType(self, node: ty.UnkType):
        pass

    def visit_VoidType(self, node: ty.NilType):
        pass

    def visit_GenericType(self, node: ty.GenericType):
        pass

    def visit_PlaceholderType(self, node: ty.PlaceholderType):
        pass

    def visit_Arg(self, node: ast.Arg):
        self.visit(node.type)
        self.visit(node.default)

    def visit_Function(self, node: ast.Function):
        for decorator in node.decorators:
            self.visit(decorator)

        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)

    def visit_Decorator(self, node: ast.Decorator):
        self.visit(node.action)  # type: ignore

    def visit_Block(self, node: ast.Block):
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_BinaryOp(self, node: ast.BinaryOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.visit(node.operand)

    def visit_VarRef(self, node: ast.VarRef):
        pass

    def visit_UVarRef(self, node: ast.UVarRef):
        pass

    def visit_UFunction(self, node: ast.UFunction):
        self.visit(node.return_type)

    def visit_UAttr(self, node: ast.UAttr):
        self.visit(node.value)

    def visit_Attribute(self, node: ast.Attribute):
        self.visit(node.value)

    def visit_If(self, node: ast.If):
        self.visit(node.cond)
        if node.then_branch is not None:
            self.visit(node.then_branch)
        for cond, block in node.elif_branches:
            self.visit(cond)
            self.visit(block)
        if node.else_branch is not None:
            self.visit(node.else_branch)

    def visit_Return(self, node: ast.Return):
        self.visit(node.value)

    def visit_File(self, node: ast.File):
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.target)
        self.visit(node.value)

    def visit_Class(self, node: ast.Class):
        for stmt in node.body:
            self.visit(stmt)

    def visit_DocString(self, node: ast.DocString):
        pass

    def visit_Select(self, node: ast.Select):
        self.visit(node.cond)
        self.visit(node.then_expr)
        self.visit(node.else_expr)

    def visit_Guard(self, node: ast.Guard):
        self.visit(node.header)
        self.visit(node.body)

    def visit_CompositeType(self, node: ty.CompositeType):
        for param in node.params:
            self.visit(param)

    def visit_LispCall(self, node: ast.LispCall):
        for arg in node.args:
            self.visit(arg)

    def visit_NilType(self, node: ty.NilType):
        pass

    def visit_LispType_(self, node: ty.LispType_):
        pass


class IRMutator:
    def visit(self, node: ast.Node | ty.Type | str | None | list | tuple):
        if node is None:
            return
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        logger.debug(f"Visiting {node.__class__.__name__}: {node}")
        return visitor(node)

    def generic_visit(self, node: ast.Node | ty.Type | str | list | tuple):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_tuple(self, node: tuple):
        return tuple([self.visit(_) for _ in node])

    def visit_list(self, node: list):
        return [self.visit(_) for _ in node]

    def visit_FileName(self, node: ast.FileName):
        return node

    def visit_UVarRef(self, node: ast.UVarRef):
        return node

    def visit_Attribute(self, node: ast.Attribute):
        node.value = self.visit(node.value)
        return node

    def visit_UAttr(self, node: ast.UAttr):
        node.value = self.visit(node.value)
        return node

    def visit_Select(self, node: ast.Select):
        node.cond = self.visit(node.cond)
        node.then_expr = self.visit(node.then_expr)
        node.else_expr = self.visit(node.else_expr)
        return node

    def visit_CallParam(self, node: ast.CallParam):
        node.value = self.visit(node.value)
        return node

    def visit_VarDecl(self, node: ast.VarDecl):
        node.init = self.visit(node.init)
        node.decorators = tuple([self.visit(_) for _ in node.decorators])
        return node

    def visit_Arg(self, node: ast.Arg):
        node.default = self.visit(node.default)
        return node

    def visit_Literal(self, node: ast.Literal):
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_str(self, node: str):
        return node

    def visit_int(self, node: int):
        return node

    def visit_float(self, node: float):
        return node

    def visit_Function(self, node: ast.Function):
        with node.write_guard():
            node.decorators = self.visit(node.decorators)
            node.args = self.visit(node.args)
            node.body = self.visit(node.body)
        return node

    def visit_Call(self, node: ast.Call):
        with node.write_guard():
            node.target = self.visit(node.target)
            node.args = self.visit(node.args)
        return node

    def visit_UFunction(self, node: ast.UFunction):
        return node

    def visit_Decorator(self, node: ast.Decorator):
        return node

    def visit_Block(self, node: ast.Block):
        node.doc_string = self.visit(node.doc_string)
        node.stmts = tuple([self.visit(stmt) for stmt in node.stmts])
        return node

    def visit_BinaryOp(self, node: ast.BinaryOp):
        with node.write_guard():
            node.left = self.visit(node.left)
            node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node.operand = self.visit(node.operand)
        return node

    def visit_VarRef(self, node: ast.VarRef):
        return node

    def visit_If(self, node: ast.If):
        with node.write_guard():
            node.cond = self.visit(node.cond)
            node.then_branch = self.visit(node.then_branch)
            node.elif_branches = self.visit(node.elif_branches)
            node.else_branch = self.visit(node.else_branch)
        return node

    def visit_Return(self, node: ast.Return):
        node.value = self.visit(node.value)
        return node

    def visit_File(self, node: ast.File):
        node.stmts = self.visit(node.stmts)
        return node

    def visit_Assign(self, node: ast.Assign):
        with node.write_guard():
            node.target = self.visit(node.target)
            node.value = self.visit(node.value)
        return node

    def visit_Class(self, node: ast.Class):
        with node.write_guard():
            node.decorators = self.visit(node.decorators)
            node.body = self.visit(node.body)
        return node

    def visit_DocString(self, node: ast.DocString):
        return node

    def visit_Guard(self, node: ast.Guard):
        with node.write_guard():
            node.header = self.visit(node.header)
            node.body = self.visit(node.body)
        return node

    def visit_LispCall(self, node: ast.LispCall):
        with node.write_guard():
            node.args = self.visit(node.args)
        return node
