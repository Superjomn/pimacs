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
        logging.warning(f"Visiting {node}")
        self.print_indent()
        self.print(f"var {node.name}")
        if node.type is not None:
            self.print(" :")
            self.visit(node.type)
        if node.init is not None:
            self.print(" = ")
            self.visit(node.init)
        self.print("\n")

    def visit_Constant(self, node: ir.Constant):
        self.print(str(node.value))

    def visit_Type(self, node: ir.Type):
        self.print(str(node))
