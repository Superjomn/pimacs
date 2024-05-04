import logging
from dataclasses import dataclass
from typing import List

import pimacs.lang.ir as ir
import pimacs.lang.type as _ty
from pimacs.lang.ir_visitor import IRVisitor

from .context import ModuleCtx, Symbol, SymbolItem, SymbolTable


def report_sema_error(node:ir.IrNode, message:str):
    logging.error(f"Error at {node.loc}:\n{message}\n")

def catch_sema_error(cond:bool, node:ir.IrNode, message:str) -> bool:
    if cond:
        node.sema_failed = True
        report_sema_error(node, message)
        return True
    return False


class Sema(IRVisitor):
    @dataclass
    class Error:
        node: ir.IrNode
        message: str


    def __init__(self, ctx:ModuleCtx) -> None:
        self.ctx = ctx
        self.errors : List[Sema.Error] = []
        self.symbol_table = SymbolTable()

    def __call__(self, ast):
        self.ir.visit(ast)
        return self.ir

    def _add_error(self, node:ir.IrNode, message:str):
        self.errors.append(Sema.Error(node, message))

    def visit_VarDecl(self, node: ir.VarDecl):
        # deduce type
        if node.type is None:
            if node.init is not None:
                node.type = node.init.get_type()
        elif node.init is not None:
            if not self.is_type_compatible(node.type, node.init.get_type()):
                self._add_error(node, f"Cannot assign {node.init.get_type()} to {node.type}")
                node.sema_failed = True
                return

    def visit_AssignStmt(self, node: ir.AssignStmt):
        super().visit_AssignStmt(node)
        assert isinstance(node.target, ir.VarRef)
        assert isinstance(node.target.decl, (ir.VarDecl, ir.ArgDecl))

        if node.value.sema_failed or node.target.sema_failed:
            node.sema_failed = True
            return

        # check if the target is mutable
        if isinstance(node.target.decl, ir.ArgDecl):
            pass
        elif isinstance(node.target.decl, ir.VarDecl):
            if not node.target.decl.mutable:
                self._add_error(node, f"Cannot assign to immutable variable {node.target.decl.name}")
        elif isinstance(node.target.decl, ir.FuncDecl):
            self._add_error(node, f"Cannot assign to function {node.target.decl.name}")
        elif isinstance(node.target.decl, ir.ClassDef):
            self._add_error(node, f"Cannot assign to class {node.target.decl.name}")

        # check if the types are compatible
        if isinstance(node.target.decl, (ir.VarDecl, ir.ArgDecl)):
            if node.target.decl.type is None:
                node.target.decl.type = node.value.get_type()
            elif not self.is_type_compatible(node.target.decl.type, node.value.get_type()):
                self._add_error(node, f"Cannot assign {node.value.get_type()} to {node.target.decl.type}")
                node.sema_failed = True
                return

    def visit_BinaryOp(self, node: ir.BinaryOp):
        super().visit_BinaryOp(node)
        # pollute sema
        if node.left.sema_failed or node.right.sema_failed:
            node.sema_failed = True
            return

        op_to_op_check = {
            ir.BinaryOperator.ADD: self.can_type_add,
            ir.BinaryOperator.SUB: self.can_type_sub,
            ir.BinaryOperator.MUL: self.can_type_mul,
            ir.BinaryOperator.DIV: self.can_type_div,
            ir.BinaryOperator.EQ: self.can_type_eq,
            ir.BinaryOperator.NE: self.can_type_neq,
        }

        for op, check in op_to_op_check.items():
            if node.op is op:
                if not check(node.left, node.right):
                    self._add_error(node, f"Cannot {op.name} {node.left.get_type()} and {node.right.get_type()}")
                    node.sema_failed = True
                    return

        # deduce the type
        if node.left.get_type() is node.right.get_type():
            node.type = node.left.get_type()
        else:
            type_pair = (node.left.get_type(), node.right.get_type())
            tgt_type = ir.BinaryOp.type_conversions.get(type_pair, None)
            if tgt_type is None:
                self._add_error(node, f"Cannot convert {type_pair[0]} and {type_pair[1]}")
                node.sema_failed = True
                return
            else:
                node.type = tgt_type


    # TODO: Support the case where __add__ or other slots are defined

    def can_type_add(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(right.get_type())

    def can_type_sub(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(right.get_type())

    def can_type_mul(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(right.get_type())

    def can_type_div(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(right.get_type())

    def can_type_eq(self, left, right):
        return True

    def can_type_neq(self, left, right):
        return True

    def is_type_numeric(self, type: _ty.Type):
        return type is _ty.Int or type is _ty.Float

    def is_type_compatible(self, left: _ty.Type, right: _ty.Type):
        '''
        There are several cases:
        1. numeric types
        2. Optional types
        '''
        # TODO: Implement this
        return True
