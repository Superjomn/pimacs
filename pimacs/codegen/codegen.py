from pimacs.ast.ast_printer import PrinterBase


class Codegen(PrinterBase):
    def __init__(self, os):
        super().__init__(os)

    def visit_list(self, nodes):
        for node in nodes:
            self.visit(node)

    def visit_Module(self, node):
        self.visit(node.body)

    def visit_Literal(self, node):
        self.write(node.value)

    def visit_VarDecl(self, node):
        self.write(node.name)
        self.write(' = ')
        self.visit(node.value)
        self.write('\n')
