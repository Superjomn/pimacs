import logging

import pimacs.lang.ir as ir


class IRVisitor:
    def visit(self, node: ir.IrNode):
        if node is None:
            return
        logging.warning(f"Visiting {node.__class__.__name__}: {node}")
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ir.IrNode):
        raise Exception(f"No visit_{node.__class__.__name__} method")

    def visit_FileName(self, node: ir.FileName):
        return node

    def visit_File(self, node: ir.File):
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_VarDecl(self, node: ir.VarDecl):
        self.visit(node.type)
        self.visit(node.init)

    def visit_Constant(self, node: ir.Constant):
        self.visit(node.value)

    def visit_int(self, node: int):
        pass

    def visit_float(self, node: float):
        pass

    def visit_Type(self, node: ir.Type):
        pass

    def visit_FuncDecl(self, node: ir.FuncDecl):
        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)

    def visit_Block(self, node: ir.Block):
        for stmt in node.stmts:
            self.visit(stmt)

    def visit_BinaryOp(self, node: ir.BinaryOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_VarRef(self, node: ir.VarRef):
        pass

    def visit_IfStmt(self, node: ir.IfStmt):
        self.visit(node.cond)
        if node.then_branch is not None:
            self.visit(node.then_branch)
        if node.else_branch is not None:
            self.visit(node.else_branch)

    def visit_ReturnStmt(self, node: ir.ReturnStmt):
        self.visit(node.value)

    def visitFile(self, node: ir.File):
        for stmt in node.stmts:
            self.visit(stmt)


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

    def print_indent(self) -> None:
        self.os.write(' ' * self._indent * self.indent_width)

    def print(self, s: str) -> None:
        self.os.write(s)

    def indent(self) -> None:
        self._indent += 1

    def deindent(self) -> None:
        self._indent -= 1

    def visit_VarDecl(self, node: ir.VarDecl):
        self.print(f"var {node.name}")
        if node.type is not None:
            self.print(" :")
            self.visit(node.type)
        if node.init is not None:
            self.print(" = ")
            self.visit(node.init)

    def visit_Constant(self, node: ir.Constant):
        self.print(str(node.value))

    def visit_Type(self, node: ir.Type):
        self.print(str(node))

    def visit_FuncDecl(self, node: ir.FuncDecl):
        self.print(f"def {node.name} (")
        if node.args:
            print(f"args: {node.args}")
            for i, arg in enumerate(node.args):
                if i > 0:
                    self.print(", ")
                self.visit(arg)
        self.print(")")

        if node.return_type is not None:
            self.print(" -> ")
            self.visit(node.return_type)

        self.print(":\n")

        self.visit(node.body)

        self.print('\n\n')

    def visit_ArgDecl(self, node: ir.ArgDecl):
        self.print(f"{node.name}")
        if node.type is not None:
            self.print(" :")
            self.visit(node.type)
        if node.default is not None:
            self.print(" = ")
            self.visit(node.default)

    def visit_FuncCall(self, node: ir.FuncCall):
        self.print(f"{node.func}(")
        if node.args:
            for i, arg in enumerate(node.args):
                if i > 0:
                    self.print(", ")
                self.visit(arg)
        self.print(")")

    def visit_Block(self, node: ir.Block):
        self.indent()
        for stmt in node.stmts:
            self.print_indent()
            self.visit(stmt)
            self.print("\n")
        self.deindent()

    def visit_BinaryOp(self, node: ir.BinaryOp):
        self.visit(node.left)
        self.print(f" {node.op.value} ")
        self.visit(node.right)

    def visit_VarRef(self, node: ir.VarRef):
        if node.name is not None:
            self.print(node.name)
        elif node.decl is not None:
            self.print(node.decl.name)
        elif node.value is not None:
            self.visit(node.value)

    def visit_IfStmt(self, node: ir.IfStmt):
        self.print("if ")
        self.visit(node.cond)
        self.print(":\n")
        self.visit(node.then_branch)
        if node.else_branch is not None:
            self.print("else:\n")
            self.visit(node.else_branch)

    def visit_ReturnStmt(self, node: ir.ReturnStmt):
        self.print("return")
        if node.value is not None:
            self.print(" ")
            self.visit(node.value)

    def visit_File(self, node: ir.File):
        for stmt in node.body:
            self.visit(stmt)
            if not isinstance(stmt, ir.FuncDecl):
                self.print("\n")
