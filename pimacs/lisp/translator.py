from typing import *

import pimacs.lang.ir as ir
import pimacs.lisp.ir as lir
from pimacs.lang.context import SymbolTable
from pimacs.lang.ir_visitor import IRMutator, IRVisitor

'''
This file defines a translator that translates the IR to Lisp IR.
'''

class LispTranslator(IRVisitor):
    def __init__(self):
        self.lir = lir.Module()
        self.symbol_table = SymbolTable()

    def __call__(self, ir) -> lir.Module:
        self.visit(ir)
        return self.lir


    def visit_VarDecl(self, node:ir.VarDecl):
        self.symbol_table.add(node.name, node)
        ret = lir.Var(name=node.name, loc=node.loc)
        ret.loc = node.loc
        return ret

    def visit_Block(self, node:ir.Block):
        ret = lir.Block(loc=node.loc)
        for stmt in node.stmts:
            ret.stmts.append(self.visit(stmt))
        return

class IrBuilder:

    def __init__(self, module_name:str):
        self.module = lir.Module(name=module_name, loc=None)

    def set_insert_point(self, block:lir.Block):
        self.insert_point = block

    def setq(self, var:lir.Var, value:lir.Expr, loc:ir.Location | None = None):
        node = lir.SetqStmt(target=var, value=value, loc=loc)
        self.insert_point.append(node)
        return node
