import logging
from contextlib import contextmanager

import pimacs.ast.ast as ast
import pimacs.ast.type as _type

from .ast import AnalyzedClass


class IRVisitor:
    def visit(self, node: ast.Node | _type.Type | str | None):
        if node is None:
            return
        if node is str:
            return node
        logging.debug(f"Visiting {node.__class__.__name__}: {node}")
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.Node | _type.Type | str):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_AnalyzedClass(self, node):
        pass

    def visit_FileName(self, node: ast.FileName):
        return node

    def visit_VarDecl(self, node: ast.VarDecl):
        self.visit(node.type)
        for decorator in node.decorators:
            self.visit(decorator)
        self.visit(node.init)

    def visit_Constant(self, node: ast.Constant):
        if node.value is not None:
            self.visit(node.value)

    def visit_int(self, node: int):
        pass

    def visit_float(self, node: float):
        pass

    def visit_Type(self, node: ast.Type):
        pass

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
        self.visit(node.value)

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

    def visit_ListType(self, node: _type.ListType):
        for inner_type in node.inner_types:
            self.visit(inner_type)

    def visit_DictType(self, node: _type.DictType):
        self.visit(node.key_type)
        self.visit(node.value_type)

    def visit_SetType(self, node: _type.SetType):
        for inner_type in node.inner_types:
            self.visit(inner_type)


class IRMutator:
    def visit(self, node: ast.Node | _type.Type | str | None):
        if node is None:
            return
        logging.debug(f"Visiting {node.__class__.__name__}: {node}")
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.Node | _type.Type | str):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_FileName(self, node: ast.FileName):
        return node

    def visit_UVarRef(self, node: ast.UVarRef):
        node.target_type = self.visit(node.target_type)
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
        node.type = self.visit(node.type)
        node.init = self.visit(node.init)
        node.decorators = tuple([self.visit(_) for _ in node.decorators])
        return node

    def visit_Arg(self, node: ast.Arg):
        node.type = self.visit(node.type)
        node.default = self.visit(node.default)
        return node

    def visit_Constant(self, node: ast.Constant):
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_str(self, node: str):
        return node

    def visit_int(self, node: int):
        return node

    def visit_float(self, node: float):
        return node

    def visit_Type(self, node: ast.Type):
        return node

    def visit_Function(self, node: ast.Function):
        decorators = []
        for i, decorator in enumerate(node.decorators):
            decorators.append(self.visit(decorator))
        node.decorators = tuple(decorators)

        args = []
        for i, arg in enumerate(node.args):
            args.append(self.visit(arg))
        node.args = tuple(args)

        node.body = self.visit(node.body)
        return node

    def visit_Call(self, node: ast.Call):
        node.func = self.visit(node.func)
        node.args = tuple([self.visit(_) for _ in node.args])
        node.type_spec = tuple([self.visit(_) for _ in node.type_spec])
        return node

    def visit_UFunction(self, node: ast.UFunction):
        node.return_type = self.visit(node.return_type)
        return node

    def visit_Decorator(self, node: ast.Decorator):
        return node

    def visit_Block(self, node: ast.Block):
        node.doc_string = self.visit(node.doc_string)
        node.stmts = tuple([self.visit(stmt) for stmt in node.stmts])
        return node

    def visit_BinaryOp(self, node: ast.BinaryOp):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node.value = self.visit(node.value)
        return node

    def visit_VarRef(self, node: ast.VarRef):
        return node

    def visit_If(self, node: ast.If):
        node.cond = self.visit(node.cond)
        node.then_branch = self.visit(node.then_branch)
        elif_branches = []
        for cond, block in node.elif_branches:
            elif_branches.append((self.visit(cond), self.visit(block)))

        node.else_branch = self.visit(node.else_branch)
        return node

    def visit_Return(self, node: ast.Return):
        node.value = self.visit(node.value)
        return node

    def visit_File(self, node: ast.File):
        stmts = []
        for i, stmt in enumerate(node.stmts):
            stmts.append(self.visit(stmt))
        node.stmts = tuple(stmts)

        return node

    def visit_Assign(self, node: ast.Assign):
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        return node

    def visit_Class(self, node: ast.Class):
        node.decorators = tuple([self.visit(_) for _ in node.decorators])
        node.body = tuple([self.visit(_) for _ in node.body])
        return node

    def visit_DocString(self, node: ast.DocString):
        return node

    def visit_Guard(self, node: ast.Guard):
        node.header = self.visit(node.header)
        node.body = self.visit(node.body)
        return node

    def visit_ListType(self, node: _type.ListType):
        node.inner_types = tuple(self.visit(_) for _ in node.inner_types)
        return node

    def visit_DictType(self, node: _type.DictType):
        node.key_type = self.visit(node.key_type)
        node.value_type = self.visit(node.value_type)
        return node

    def visit_SetType(self, node: _type.SetType):
        node.inner_types = tuple(self.visit(_) for _ in node.inner_types)
        return node

    def visit_AnalyzedClass(self, node: AnalyzedClass):
        return self.visit_Class(node)


class StringStream:
    def __init__(self) -> None:
        self.s = ""

    def write(self, s: str) -> None:
        self.s += s


class IRPrinter(IRVisitor):
    indent_width = 4

    def __init__(self, os) -> None:
        self.os = os
        self._indent: int = 0

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
            self.visit(node.type)
        if node.init is not None:
            self.put(" = ")
            self.visit(node.init)

    def visit_Constant(self, node: ast.Constant):
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
        if isinstance(node.func, str):
            self.put(f"{node.func}")
        elif isinstance(node.func, ast.VarRef):
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ast.Function):
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ast.UFunction):
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ast.Class):
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ast.Arg):
            # for cls(...) call
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ast.UVarRef):
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ast.VarDecl):
            assert node.func.init is not None
            self.visit(node.func.init)
        else:
            raise Exception(f"{node.loc}\nInvalid function call: {
                            node.func}, type: {type(node.func)}")

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
        self.visit(node.value)
        self.put(")")

    def visit_VarRef(self, node: ast.VarRef):
        if node.name:
            self.put(node.name)
        elif node.target is not None:
            self.put(node.target.name)

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
            self.put(f"@{node.action.func.name}(")  # type: ignore
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

    def visit_AnanlyzedClass(self, node: AnalyzedClass):
        self.visit_Class(node)

    def visit_UVarRef(self, node: ast.UVarRef):
        self.put(f"{node.name}")

    def visit_UAttr(self, node: ast.UAttr):
        self.visit(node.value)
        self.put(f".{node.attr}")

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

    def visit_DictType(self, node: _type.DictType):
        self.put("{")
        self.visit(node.key_type)
        self.put(":")
        self.visit(node.value_type)
        self.put("}")

    def visit_ListType(self, node: _type.ListType):
        self.put("[")
        self.visit(node.inner_types[0])
        self.put("]")

    def visit_SetType(self, node: _type.SetType):
        self.put("{")
        self.visit(node.inner_types[0])
        self.put("}")

    @contextmanager
    def indent_guard(self):
        self.indent()
        yield
        self.deindent()
