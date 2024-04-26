import logging
from contextlib import contextmanager

import pimacs.lang.ir as ir
import pimacs.lang.type as _type


class IRVisitor:
    def visit(self, node: ir.IrNode | _type.Type | str | None):
        if node is None:
            return
        if node is str:
            return node
        logging.debug(f"Visiting {node.__class__.__name__}: {node}")
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ir.IrNode | _type.Type | str):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_FileName(self, node: ir.FileName):
        return node

    def visit_VarDecl(self, node: ir.VarDecl):
        self.visit(node.type)
        self.visit(node.init)

    def visit_Constant(self, node: ir.Constant):
        if node.value is not None:
            self.visit(node.value)

    def visit_int(self, node: int):
        pass

    def visit_float(self, node: float):
        pass

    def visit_Type(self, node: ir.Type):
        pass

    def visit_FuncDecl(self, node: ir.FuncDecl):
        for decorator in node.decorators:
            self.visit(decorator)

        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)

    def visit_Decorator(self, node: ir.Decorator):
        self.visit(node.action)

    def visit_Block(self, node: ir.Block):
        self.visit(node.doc_string)
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_BinaryOp(self, node: ir.BinaryOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ir.UnaryOp):
        self.visit(node.value)

    def visit_VarRef(self, node: ir.VarRef):
        pass

    def visit_IfStmt(self, node: ir.IfStmt):
        self.visit(node.cond)
        if node.then_branch is not None:
            self.visit(node.then_branch)
        for (cond, block) in node.elif_branches:
            self.visit(cond)
            self.visit(block)
        if node.else_branch is not None:
            self.visit(node.else_branch)

    def visit_ReturnStmt(self, node: ir.ReturnStmt):
        self.visit(node.value)

    def visit_File(self, node: ir.File):
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_AssignStmt(self, node: ir.AssignStmt):
        self.visit(node.target)
        self.visit(node.value)

    def visit_ClassDef(self, node: ir.ClassDef):
        for stmt in node.body:
            self.visit(stmt)

    def visit_LispVarRef(self, node: ir.LispVarRef):
        pass

    def visit_DocString(self, node: ir.DocString):
        pass

    def visit_SelectExpr(self, node: ir.SelectExpr):
        self.visit(node.cond)
        self.visit(node.then_expr)
        self.visit(node.else_expr)

    def visit_GuardStmt(self, node: ir.GuardStmt):
        self.visit(node.header)
        self.visit(node.body)


class IRMutator:
    def visit(self, node: ir.IrNode | _type.Type | str | None):
        if node is None:
            return
        logging.debug(f"Visiting {node.__class__.__name__}: {node}")
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ir.IrNode | _type.Type | str):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_FileName(self, node: ir.FileName):
        return node

    def visit_SelectExpr(self, node: ir.SelectExpr):
        node.cond = self.visit(node.cond)
        node.then_expr = self.visit(node.then_expr)
        node.else_expr = self.visit(node.else_expr)
        return node

    def visit_CallParam(self, node: ir.CallParam):
        node.value = self.visit(node.value)
        return node

    def visit_VarDecl(self, node: ir.VarDecl):
        node.type = self.visit(node.type)
        node.init = self.visit(node.init)
        return node

    def visit_Constant(self, node: ir.Constant):
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_str(self, node: str):
        return node

    def visit_int(self, node: int):
        return node

    def visit_float(self, node: float):
        return node

    def visit_Type(self, node: ir.Type):
        return node

    def visit_FuncDecl(self, node: ir.FuncDecl):
        for i, decorator in enumerate(node.decorators):
            node.decorators[i] = self.visit(decorator)

        for i, arg in enumerate(node.args):
            node.args[i] = self.visit(arg)
        node.body = self.visit(node.body)
        return node

    def visit_Decorator(self, node: ir.Decorator):
        return node

    def visit_Block(self, node: ir.Block):
        node.doc_string = self.visit(node.doc_string)
        for i, stmt in enumerate(node.stmts):
            node.stmts[i] = self.visit(stmt)
        return node

    def visit_BinaryOp(self, node: ir.BinaryOp):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node: ir.UnaryOp):
        node.value = self.visit(node.value)
        return node

    def visit_VarRef(self, node: ir.VarRef):
        return node

    def visit_IfStmt(self, node: ir.IfStmt):
        node.cond = self.visit(node.cond)
        node.then_branch = self.visit(node.then_branch)
        elif_branches = []
        for cond, block in node.elif_branches:
            elif_branches.append((self.visit(cond), self.visit(block)))

        node.else_branch = self.visit(node.else_branch)
        return node

    def visit_ReturnStmt(self, node: ir.ReturnStmt):
        node.value = self.visit(node.value)
        return node

    def visit_File(self, node: ir.File):
        for i, stmt in enumerate(node.stmts):
            node.stmts[i] = self.visit(stmt)
        return node

    def visit_AssignStmt(self, node: ir.AssignStmt):
        node.target = self.visit(node.target)
        node.value = self.visit(node.value)
        return node

    def visit_ClassDef(self, node: ir.ClassDef):
        for i, stmt in enumerate(node.body):
            node.body[i] = self.visit(stmt)
        return node

    def visit_LispVarRef(self, node: ir.LispVarRef):
        return node

    def visit_DocString(self, node: ir.DocString):
        return node

    def visit_GuardStmt(self, node: ir.GuardStmt):
        node.header = self.visit(node.header)
        node.body = self.visit(node.body)
        return node


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

    def __call__(self, node: ir.IrNode) -> None:
        self.visit(node)

    def put_indent(self) -> None:
        self.os.write(' ' * self._indent * self.indent_width)

    def put(self, s: str) -> None:
        self.os.write(s)

    def indent(self) -> None:
        self._indent += 1

    def deindent(self) -> None:
        self._indent -= 1

    def visit_VarDecl(self, node: ir.VarDecl):
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

    def visit_Constant(self, node: ir.Constant):
        if node.value is not None:
            self.put(str(node.value))
        else:
            self.put("nil")

    def visit_Type(self, node: ir.Type):
        self.put(str(node))

    def visit_FuncDecl(self, node: ir.FuncDecl):
        for decorator in node.decorators:
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

    def visit_ArgDecl(self, node: ir.ArgDecl):
        self.put(f"{node.name}")
        if node.type is not None:
            self.put(" :")
            self.visit(node.type)
        if node.default is not None:
            self.put(" = ")
            self.visit(node.default)

    def visit_FuncCall(self, node: ir.FuncCall):
        if isinstance(node.func, str):
            self.put(f"{node.func}")
        elif isinstance(node.func, ir.ArgDecl):
            self.put(f"{node.func.name}")
        elif isinstance(node.func, ir.VarRef):
            self.put(f"{node.func.name}")
        else:
            raise Exception(f"{node.loc}\nInvalid function call: {node.func}")

        self.put("(")
        if node.args:
            for i, arg in enumerate(node.args):
                if i > 0:
                    self.put(", ")
                self.visit(arg)
        self.put(")")


    def visit_Block(self, node: ir.Block):
        with self.indent_guard():
            if node.doc_string is not None:
                self.put_indent()
                self.visit(node.doc_string)
                self.put("\n")
            for stmt in node.stmts:
                self.put_indent()
                self.visit(stmt)
                self.put("\n")

    def visit_BinaryOp(self, node: ir.BinaryOp):
        self.visit(node.left)
        self.put(f" {node.op.value} ")
        self.visit(node.right)

    def visit_UnaryOp(self, node: ir.UnaryOp):
        self.put("(")
        self.put(f"{node.op.value}")
        self.visit(node.value)
        self.put(")")

    def visit_VarRef(self, node: ir.VarRef):
        if node.name:
            self.put(node.name)
        elif node.decl is not None:
            self.put(node.decl.name)
        elif node.value is not None:
            self.visit(node.value)

    def visit_IfStmt(self, node: ir.IfStmt):
        self.put("if ")
        self.visit(node.cond)
        self.put(":\n")
        self.visit(node.then_branch)
        for (cond, block) in node.elif_branches:
            self.put_indent()
            self.put("elif ")
            self.visit(cond)
            self.put(":\n")
            self.visit(block)
        if node.else_branch is not None:
            self.put_indent()
            self.put("else:\n")
            self.visit(node.else_branch)

    def visit_ReturnStmt(self, node: ir.ReturnStmt):
        self.put("return")
        if node.value is not None:
            self.put(" ")
            self.visit(node.value)

    def visit_File(self, node: ir.File):
        for stmt in node.stmts:
            if isinstance(stmt, ir.FuncDecl):
                self.put("\n")
            self.put_indent()
            self.visit(stmt)
            self.put("\n")

    def visit_Decorator(self, node: ir.Decorator):
        if isinstance(node.action, ir.FuncCall):
            self.put(f"@{node.action.func}(")
            if node.action.args:
                for i, arg in enumerate(node.action.args):
                    if i > 0:
                        self.put(", ")
                    self.visit(arg)
            self.put(")")
        elif isinstance(node.action, str):
            self.put(f"@{node.action}")
        else:
            raise Exception(f"Invalid decorator action: {node.action}")

    def visit_AssignStmt(self, node: ir.AssignStmt):
        self.visit(node.target)
        self.put(" = ")
        self.visit(node.value)

    def visit_ClassDef(self, node: ir.ClassDef):
        self.put(f"class {node.name}:\n")
        with self.indent_guard():
            for stmt in node.body:
                if isinstance(stmt, ir.FuncDecl):
                    self.put("\n")
                self.put_indent()
                self.visit(stmt)
                self.put("\n")

    def visit_LispVarRef(self, node: ir.LispVarRef):
        self.put(f"%{node.name}")

    def visit_DocString(self, node: ir.DocString):
        self.put(f'"{node.content}"')

    def visit_SelectExpr(self, node: ir.SelectExpr):
        self.visit(node.then_expr)
        self.put(" if ")
        self.visit(node.cond)
        self.put(" else ")
        self.visit(node.else_expr)

    def visit_GuardStmt(self, node: ir.GuardStmt):
        self.put("guard ")
        self.visit(node.header)
        self.put(":\n")
        self.visit(node.body)

    def visit_CallParam(self, node: ir.CallParam):
        self.visit(node.value)

    @contextmanager
    def indent_guard(self):
        self.indent()
        yield
        self.deindent()
