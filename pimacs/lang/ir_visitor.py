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
        for decorator in node.decorators:
            self.visit(decorator)

        for arg in node.args:
            self.visit(arg)
        self.visit(node.body)

    def visit_Decorator(self, node: ir.Decorator):
        self.visit(node.action)

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

    def put(self, s: str) -> None:
        self.os.write(s)

    def indent(self) -> None:
        self._indent += 1

    def deindent(self) -> None:
        self._indent -= 1

    def visit_VarDecl(self, node: ir.VarDecl):
        self.put(f"var {node.name}")
        if node.type is not None:
            self.put(" :")
            self.visit(node.type)
        if node.init is not None:
            self.put(" = ")
            self.visit(node.init)

    def visit_Constant(self, node: ir.Constant):
        self.put(str(node.value))

    def visit_Type(self, node: ir.Type):
        self.put(str(node))

    def visit_FuncDecl(self, node: ir.FuncDecl):
        for decorator in node.decorators:
            self.visit(decorator)
            self.put("\n")

        self.put(f"def {node.name} (")
        if node.args:
            print(f"args: {node.args}")
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

        self.put('\n\n')

    def visit_ArgDecl(self, node: ir.ArgDecl):
        self.put(f"{node.name}")
        if node.type is not None:
            self.put(" :")
            self.visit(node.type)
        if node.default is not None:
            self.put(" = ")
            self.visit(node.default)

    def visit_FuncCall(self, node: ir.FuncCall):
        self.put(f"{node.func}(")
        if node.args:
            for i, arg in enumerate(node.args):
                if i > 0:
                    self.put(", ")
                self.visit(arg)
        self.put(")")

    def visit_Block(self, node: ir.Block):
        self.indent()
        for stmt in node.stmts:
            self.print_indent()
            self.visit(stmt)
            self.put("\n")
        self.deindent()

    def visit_BinaryOp(self, node: ir.BinaryOp):
        self.visit(node.left)
        self.put(f" {node.op.value} ")
        self.visit(node.right)

    def visit_VarRef(self, node: ir.VarRef):
        if node.name is not None:
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
        if node.else_branch is not None:
            self.put("else:\n")
            self.visit(node.else_branch)

    def visit_ReturnStmt(self, node: ir.ReturnStmt):
        self.put("return")
        if node.value is not None:
            self.put(" ")
            self.visit(node.value)

    def visit_File(self, node: ir.File):
        for stmt in node.body:
            self.visit(stmt)
            if not isinstance(stmt, ir.FuncDecl):
                self.put("\n")

    def visit_Decorator(self, node: ir.Decorator):
        self.print_indent()
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
        self.indent()
        for stmt in node.body:
            self.print_indent()
            self.visit(stmt)
            if not isinstance(stmt, ir.FuncDecl):
                self.put("\n")
        self.deindent()

    def visit_LispVarRef(self, node: ir.LispVarRef):
        self.put(f"%{node.name}")
