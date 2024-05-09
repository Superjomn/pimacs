import logging
from dataclasses import dataclass
from typing import List

import pimacs.lang.ir as ir
import pimacs.lang.type as _ty
from pimacs.lang.ir_visitor import IRMutator, IRVisitor

from .context import ModuleCtx, Scope, Symbol, SymbolItem, SymbolTable


def report_sema_error(node: ir.IrNode, message: str):
    logging.error(f"Error at {node.loc}:\n{message}\n")


def catch_sema_error(cond: bool, node: ir.IrNode, message: str) -> bool:
    if cond:
        node.sema_failed = True
        report_sema_error(node, message)
        return True
    return False


def any_failed(*nodes: ir.IrNode) -> bool:
    return any(node.sema_failed for node in nodes)


def can_assign_type(target: _ty.Type, source: _ty.Type | List[_ty.Type]):
    """
    There are several cases:
    1. numeric types
    2. Optional types
    """
    return True


def is_type_numeric(type: _ty.Type):
    return type is _ty.Int or type is _ty.Float


def is_type_compatible(left: _ty.Type, right: _ty.Type):
    """
    a + b; there might be __add__ method with overloading
    """
    return True


def get_common_type(left: _ty.Type, right: _ty.Type):
    return left


class Sema(IRMutator):
    """
    Clean up the symbols in the IR, like unify the VarRef or FuncDecl with the same name in the same scope.
    """

    # NOTE, the ir nodes should be created when visited.

    def __init__(self):
        self.sym_tbl = SymbolTable()
        self._succeeded = True

    def report_error(self, node: ir.IrNode, message: str):
        report_sema_error(node, message)
        node.sema_failed = True
        self._succeeded = False

    def visit_UnresolvedVarRef(self, node: ir.UnresolvedVarRef):
        node = super().visit_UnresolvedVarRef(node)
        if node.name.startswith("self."):
            # deal with members
            member = self.sym_tbl.get_symbol(
                name=node.name[5:], kind=Symbol.Kind.Member
            )
            if member is None:
                raise KeyError(f"{node.loc}\nMember {node.name} is not declared")
            var = ir.VarRef(decl=member, loc=node.loc)  # type: ignore

            obj = self.sym_tbl.get_symbol(name="self", kind=Symbol.Kind.Arg)
            if not obj:
                obj = ir.UnresolvedVarRef(name="self", loc=node.loc, target_type=_type.Unk)  # type: ignore
                catch_sema_error(True, node, f"`self` is not declared")
            return ir.MemberRef(obj=ir.VarRef(decl=obj, loc=node.loc), member=var, loc=node.loc)  # type: ignore

        elif sym := self.sym_tbl.get_symbol(
            name=node.name, kind=[Symbol.Kind.Var, Symbol.Kind.Arg]
        ):
            if isinstance(sym, ir.VarDecl):
                var = ir.VarRef(decl=sym, loc=node.loc)  # type: ignore
            elif isinstance(sym, ir.ArgDecl):
                var = ir.VarRef(decl=sym, loc=node.loc)
            elif isinstance(sym, ir.VarRef):
                var = sym
            else:
                raise ValueError(f"{node.loc}\nUnknown symbol type {sym}")
            return var
        else:
            node.sema_failed = True
            report_sema_error(node, f"Symbol {node.name} not found")
            return node

    def visit_VarDecl(self, node: ir.VarDecl):
        node = super().visit_VarDecl(node)
        if self.sym_tbl.current_scope.kind is Scope.Kind.Class:
            # Declare a member
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Member)
            self.sym_tbl.add_symbol(symbol, node)
        else:
            # Declare a local variable
            self.sym_tbl.add_symbol(Symbol(name=node.name, kind=Symbol.Kind.Var), node)

        return self.verify_VarDecl(node)

    def verify_VarDecl(self, node: ir.VarDecl) -> ir.VarDecl:
        super().visit_VarDecl(node)
        if node.sema_failed:
            return node
        if not node.init:
            if (not node.type) or (node.type is _ty.Unk):
                self.report_error(
                    node, f"Variable {node.name} should have a type or an initial value"
                )
                return node
        else:
            if (not node.type) or (node.type is _ty.Unk):
                node.type = node.init.get_type()
                return node
            else:
                if not can_assign_type(node.type, node.init.get_type()):
                    self.report_error(
                        node, f"Cannot assign {node.init.get_type()} to {node.type}"
                    )
                    return node
        return node

    def visit_SelectExpr(self, node: ir.SelectExpr):
        node = super().visit_SelectExpr(node)
        if any_failed(node.cond, node.then_expr, node.else_expr):
            node.sema_failed = True
            return node

        return self.verify_SelectExpr(node)

    def verify_SelectExpr(self, node: ir.SelectExpr) -> ir.SelectExpr:
        if node.sema_failed:
            return node
        if not is_type_compatible(node.then_expr.get_type(), node.else_expr.get_type()):
            self.report_error(
                node,
                f"Cannot convert {node.then_expr.get_type()} and {node.else_expr.get_type()}",
            )
            return node
        return node

    def visit_FuncDecl(self, node: ir.FuncDecl):
        node = super().visit_FuncDecl(node)
        within_class = self.sym_tbl.current_scope.kind is Scope.Kind.Class
        with self.sym_tbl.scope_guard(kind=Scope.Kind.Func):
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Func)
            # TODO[Superjomn]: support function overloading
            if self.sym_tbl.contains_locally(symbol):
                self.report_error(node, f"Function {node.name} already exists")
            args = node.args if node.args else []
            if within_class:
                # This function is a member function
                if node.is_classmethod:
                    cls_arg = args[0]  # cls placeholder
                    if cls_arg.name != "cls":
                        self.report_error(
                            node, f"Class method should have the first arg named 'cls'"
                        )
                    cls_arg.kind = ir.ArgDecl.Kind.cls_placeholder
                elif not node.is_staticmethod:
                    self_arg = args[0]  # self placeholder
                    if self_arg.name != "self":
                        self.report_error(
                            node, f"Method should have the first arg named 'self'"
                        )
                    self_arg.kind = ir.ArgDecl.Kind.self_placeholder

            args = [self.visit(arg) for arg in args]
            body = self.visit(node.body)
            return_type = self.visit(node.return_type)
            decorators = [self.visit(decorator) for decorator in node.decorators]

            new_node = ir.FuncDecl(
                name=node.name,
                args=args,
                body=body,
                return_type=return_type,
                loc=node.loc,
                decorators=decorators,
            )
            if any_failed(*args, body, *decorators):
                new_node.sema_failed = True

            new_node = self.sym_tbl.add_symbol(symbol, new_node)
            return self.verify_FuncDecl(new_node)

    def verify_FuncDecl(self, node: ir.FuncDecl) -> ir.FuncDecl:
        if node.sema_failed:
            return node

        # check if argument names are unique
        arg_names = [arg.name for arg in node.args]
        if len(arg_names) != len(set(arg_names)):
            self.report_error(node, f"Argument names should be unique, got {arg_names}")
        # check if argument types are valid
        for arg in node.args:
            if arg.sema_failed:
                continue
            if not arg.type or arg.type is _ty.Unk:
                self.report_error(arg, f"Argument {arg.name} should have a type")
        # check if return type is valid
        if not node.return_type or node.return_type is _ty.Unk:
            node.return_type = _ty.Nil
            if node.body.return_type is not _ty.Nil:
                self.report_error(
                    node,
                    f"Function which returns non-nil values should set return type",
                )
        elif node.body.return_type and node.body.return_type is not _ty.Nil:
            if not can_assign_type(node.return_type, node.body.return_type):
                self.report_error(
                    node,
                    f"Cannot convert {node.body.return_type} to {node.return_type}",
                )

        return node

    def visit_ClassDef(self, node: ir.ClassDef):
        node = super().visit_ClassDef(node)
        if self.sym_tbl.current_scope.kind is not Scope.Kind.Global:
            self.report_error(node, f"Class should be declared in the global scope")

        with self.sym_tbl.scope_guard(kind=Scope.Kind.Class):
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Class)
            if self.sym_tbl.contains_locally(symbol):
                self.report_error(node, f"Class {node.name} already exists")

            body = [self.visit(stmt) for stmt in node.body]

            node = ir.ClassDef(name=node.name, body=body, loc=node.loc)
            node = self.sym_tbl.add_symbol(symbol, node)
            return self.verify_ClassDef(node)

    def verify_ClassDef(self, node: ir.ClassDef) -> ir.ClassDef:
        if node.sema_failed:
            return node
        return node

    def visit_FuncCall(self, node: ir.FuncCall):
        node = super().visit_FuncCall(node)
        print(f"*** visit_FuncCall: {node}")
        with self.sym_tbl.scope_guard():
            func_name = node.func
            assert isinstance(func_name, str)
            func = self.sym_tbl.get_symbol(
                name=func_name,
                kind=[
                    Symbol.Kind.Func,
                    Symbol.Kind.Arg,
                    Symbol.Kind.Var,
                    Symbol.Kind.Class,
                ],
            )
            if not func:
                self.report_error(node, f"Target function/class {func_name} not found")
                return self.verify_FuncCall(node, func)
            elif isinstance(func, ir.VarDecl):
                true_func = Sema.get_true_var(func)
                if isinstance(true_func, ir.LispVarRef):
                    if isinstance(true_func, ir.LispVarRef):
                        return self.verify_FuncCall(
                            ir.LispFuncCall(
                                func=true_func, args=node.args, loc=node.loc
                            ),
                            func,
                        )
                self.report_error(
                    node, f"{node.loc}\nUnknown lisp function call {func}"
                )
                return node
            else:
                assert func is not None
                node = ir.FuncCall(func=func, args=node.args, loc=node.loc)  # type: ignore
                return self.verify_FuncCall(node, func)

    def verify_FuncCall(
        self,
        node: ir.FuncCall | ir.LispFuncCall,
        func: ir.VarDecl | ir.LispVarRef | None,
    ) -> ir.FuncCall | ir.LispFuncCall:
        if node.sema_failed:
            return node

        if any_failed(*node.args):
            node.sema_failed = True
        if isinstance(node, ir.FuncCall):
            # TODO[superjomn]: check if the number of arguments and type is correct
            return node
        return node

    def visit_LispFuncCall(self, node: ir.LispFuncCall):
        return super().visit_LispFuncCall(node)

    def visit_Block(self, node: ir.Block):
        node = super().visit_Block(node)
        with self.sym_tbl.scope_guard():
            if any_failed(*node.stmts):
                node.sema_failed = True
        return self.verify_Block(node)

    def verify_Block(self, node: ir.Block) -> ir.Block:
        if node.sema_failed:
            return node

        return_types = set()
        # gather return types
        for return_stmt in node.stmts:
            if isinstance(return_stmt, ir.ReturnStmt):
                if return_stmt.value:
                    return_types.add(return_stmt.value.get_type())
        node.return_type = list(return_types) if return_types else [_ty.Nil]
        return node

    def visit_ArgDecl(self, node: ir.ArgDecl):
        node = super().visit_ArgDecl(node)
        assert (
            self.sym_tbl.current_scope.kind is Scope.Kind.Func
        ), f"{node.loc}\nArgDecl should be in a function"
        symbol = Symbol(name=node.name, kind=Symbol.Kind.Arg)
        node = self.sym_tbl.add_symbol(symbol, node)
        return self.verify_ArgDecl(node)

    def verify_ArgDecl(self, node: ir.ArgDecl) -> ir.ArgDecl:
        if not (node.is_cls_placeholder or node.is_self_placeholder):
            if node.type is _ty.Unk and node.default is None:
                self.report_error(
                    node,
                    f"{node.loc}\nArg {node.name} should have a type or a default value",
                )
            elif node.default is not None and node.type is not _ty.Unk:
                if node.default.get_type() is not node.type:
                    self.report_error(
                        node,
                        f"{node.loc}\nCannot assign {node.default.get_type()} to {node.type}",
                    )
        return node

    def visit_BinaryOp(self, node: ir.BinaryOp):
        node = super().visit_BinaryOp(node)
        return self.verify_BinaryOp(node)

    def verify_BinaryOp(self, node: ir.BinaryOp) -> ir.BinaryOp:
        if any_failed(node.left, node.right):
            node.sema_failed = True
            return node

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
                    self.report_error(
                        node,
                        f"Cannot {op.name} {node.left.get_type()} and {node.right.get_type()}",
                    )
                    node.sema_failed = True
                    return node

        if not is_type_compatible(node.left.get_type(), node.right.get_type()):
            self.report_error(
                node,
                f"Cannot convert {node.left.get_type()} and {node.right.get_type()}",
            )

        node.type = get_common_type(node.left.get_type(), node.right.get_type())
        return node

    def can_type_add(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(
            right.get_type()
        )

    def can_type_sub(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(
            right.get_type()
        )

    def can_type_mul(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(
            right.get_type()
        )

    def can_type_div(self, left, right):
        return self.is_type_numeric(left.get_type()) and self.is_type_numeric(
            right.get_type()
        )

    def can_type_eq(self, left, right):
        return True

    def can_type_neq(self, left, right):
        return True

    def visit_UnaryOp(self, node: ir.UnaryOp):
        node = super().visit_UnaryOp(node)
        return node

    def verify_UnaryOp(self, node: ir.UnaryOp) -> ir.UnaryOp:
        if node.value.sema_failed:
            node.sema_failed = True
            return node
        return node

    def visit_CallParam(self, node: ir.CallParam):
        node = super().visit_CallParam(node)
        return node

    def verify_CallParam(self, node: ir.CallParam) -> ir.CallParam:
        if node.sema_failed:
            return node
        return node

    @staticmethod
    def get_true_var(
        node: ir.VarDecl | ir.VarRef,
    ) -> ir.VarDecl | ir.LispVarRef | ir.ArgDecl | ir.UnresolvedVarRef | ir.Expr:
        if isinstance(node, ir.VarDecl) and node.init:
            return node.init
        if isinstance(node, ir.VarRef):
            assert node.decl
            return node.decl
        if isinstance(node, ir.LispVarRef):
            return node

        return node
