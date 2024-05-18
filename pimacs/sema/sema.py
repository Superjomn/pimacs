import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List

import pimacs.ast.ast as ast
import pimacs.ast.type as _ty
from pimacs.sema.ast_visitor import IRMutator, IRVisitor
from pimacs.sema.func import FuncOverloads

from .ast import AnalyzedClassDef
from .context import FuncSymbol, ScopeKind, Symbol, SymbolTable
from .utils import ClassId, ModuleId, bcolors, print_colored


def report_sema_error(node: ast.IrNode, message: str):
    print_colored(f"{node.loc}\n")
    print_colored(f"Error: {message}\n\n", bcolors.FAIL)


def catch_sema_error(cond: bool, node: ast.IrNode, message: str) -> bool:
    if cond:
        node.sema_failed = True
        report_sema_error(node, message)
        return True
    return False


def any_failed(*nodes: ast.IrNode) -> bool:
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

        self._cur_class: ast.ClassDef | None = None

    @property
    def succeed(self) -> bool:
        return self._succeeded

    def report_error(self, node: ast.IrNode, message: str):
        report_sema_error(node, message)
        node.sema_failed = True
        self._succeeded = False

    def visit_UnresolvedVarRef(self, node: ast.UnresolvedVarRef):
        node = super().visit_UnresolvedVarRef(node)
        if node.name.startswith("self."):
            # deal with members
            var_name = node.name[5:]
            var = ast.VarRef(decl=member, loc=node.loc)  # type: ignore

            obj = self.sym_tbl.get_symbol(name="self", kind=Symbol.Kind.Arg)
            if not obj:
                obj = ast.UnresolvedVarRef(
                    # TODO: fix the type
                    name="self", loc=node.loc, target_type=_ty.Unk)  # type: ignore
                catch_sema_error(True, node, f"`self` is not declared")
            # The Attr cannot be resolved now, it can only be resolved in bulk when a class is fully analyzed.
            return ast.UnresolvedAttr(value=obj, attr=var_name, loc=node.loc)

        elif sym := self.sym_tbl.get_symbol(
            name=node.name, kind=[Symbol.Kind.Var, Symbol.Kind.Arg]
        ):
            if isinstance(sym, ast.VarDecl):
                var = ast.VarRef(decl=sym, loc=node.loc)  # type: ignore
            elif isinstance(sym, ast.ArgDecl):
                var = ast.VarRef(decl=sym, loc=node.loc)
            elif isinstance(sym, ast.VarRef):
                var = sym
            else:
                raise ValueError(f"{node.loc}\nUnknown symbol type {sym}")
            return var
        else:
            node.sema_failed = True
            report_sema_error(node, f"Symbol {node.name} not found")
            return node

    def visit_VarDecl(self, node: ast.VarDecl):
        node = super().visit_VarDecl(node)
        if self.sym_tbl.current_scope.kind is ScopeKind.Class:
            # Declare a member
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Member)
            self.sym_tbl.insert(symbol, node)
        else:
            # Declare a local variable
            if self.sym_tbl.contains_locally(
                Symbol(name=node.name, kind=Symbol.Kind.Var)
            ):
                self.report_error(node, f"Variable {node.name} already exists")
            else:
                self.sym_tbl.insert(
                    Symbol(name=node.name, kind=Symbol.Kind.Var), node)

        return self.verify_VarDecl(node)

    def verify_VarDecl(self, node: ast.VarDecl) -> ast.VarDecl:
        if node.sema_failed:
            return node
        if not node.init:
            if (not node.type) or (node.type is _ty.Unk):
                self.report_error(
                    node, f"Variable {
                        node.name} should have a type or an initial value"
                )
                return node
        else:
            if (not node.type) or (node.type is _ty.Unk):
                node.type = node.init.get_type()
                return node
            else:
                if not can_assign_type(node.type, node.init.get_type()):
                    self.report_error(
                        node, f"Cannot assign {node.init.get_type()} to {
                            node.type}"
                    )
                    return node
        return node

    def visit_SelectExpr(self, node: ast.SelectExpr):
        node = super().visit_SelectExpr(node)
        if any_failed(node.cond, node.then_expr, node.else_expr):
            node.sema_failed = True
            return node

        return self.verify_SelectExpr(node)

    def verify_SelectExpr(self, node: ast.SelectExpr) -> ast.SelectExpr:
        if node.sema_failed:
            return node
        if not is_type_compatible(node.then_expr.get_type(), node.else_expr.get_type()):
            self.report_error(
                node,
                f"Cannot convert {node.then_expr.get_type()} and {
                    node.else_expr.get_type()}",
            )
            return node
        return node

    def visit_FuncDecl(self, node: ast.FuncDecl):
        within_class = self.sym_tbl.current_scope.kind is ScopeKind.Class
        if within_class:
            assert self._cur_class

        with self.sym_tbl.scope_guard(kind=ScopeKind.Func):
            node = super().visit_FuncDecl(node)
            symbol = FuncSymbol(node.name)
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
                    cls_arg.kind = ast.ArgDecl.Kind.cls_placeholder
                elif not node.is_staticmethod:
                    self_arg = args[0]  # self placeholder
                    if self_arg.name != "self":
                        self.report_error(
                            node, f"Method should have the first arg named 'self'"
                        )
                    self_arg.kind = ast.ArgDecl.Kind.self_placeholder

            if any_failed(*node.args, node.body, *node.decorators):
                node.sema_failed = True

        node = self.sym_tbl.insert(symbol, node)
        return self.verify_FuncDecl(node)

    def verify_FuncDecl(self, node: ast.FuncDecl) -> ast.FuncDecl:
        if node.sema_failed:
            return node

        # check if argument names are unique
        arg_names = [arg.name for arg in node.args]
        if len(arg_names) != len(set(arg_names)):
            self.report_error(
                node, f"Argument names should be unique, got {arg_names}")
        # check if argument types are valid
        for no, arg in enumerate(node.args):
            if arg.sema_failed:
                continue
            if arg.name == "self" and arg.kind is ast.ArgDecl.Kind.self_placeholder:
                pass
            elif arg.name == "cls" and arg.kind is ast.ArgDecl.Kind.cls_placeholder:
                pass
            elif not arg.type or arg.type is _ty.Unk:
                self.report_error(
                    arg, f"Argument {arg.name} should have a type")
        # check if return type is valid
        if not node.return_type or node.return_type is _ty.Unk:
            node.return_type = _ty.Nil
            return_nil = (not node.body.return_type) or (
                len(
                    node.body.return_type) == 1 and node.body.return_type[0] is _ty.Nil
            )
            if not return_nil:
                self.report_error(
                    node,
                    f"Function which returns non-nil values should set return type",
                )
        elif node.body.return_type and node.body.return_type is not _ty.Nil:
            if not can_assign_type(node.return_type, node.body.return_type):
                self.report_error(
                    node,
                    f"Cannot convert {node.body.return_type} to {
                        node.return_type}",
                )

        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        node = AnalyzedClassDef.create(node)  # type: ignore
        with node.auto_update_symbols():
            self._cur_class = node
            if self.sym_tbl.current_scope.kind is not ScopeKind.Global:
                self.report_error(
                    node, f"Class should be declared in the global scope")

            with self.sym_tbl.scope_guard(kind=ScopeKind.Class):
                node = super().visit_ClassDef(node)
                print_colored(f"** ClassDef {node.name}\n", bcolors.OKGREEN)
                self.sym_tbl.print_summary()

            symbol = Symbol(name=node.name, kind=Symbol.Kind.Class)
            if self.sym_tbl.contains_locally(symbol):
                self.report_error(node, f"Class {node.name} already exists")

            node = self.sym_tbl.insert(symbol, node)
            return self.verify_ClassDef(node)

    def verify_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if node.sema_failed:
            return node
        return node

    def visit_FuncCall(self, node: ast.FuncCall):
        node = super().visit_FuncCall(node)
        func_name = node.func
        if isinstance(func_name, ast.UnresolvedFuncDecl):
            func_name = func_name.name  # type: ignore
        assert isinstance(func_name, str)

        func = self.sym_tbl.get_function(FuncSymbol(func_name)) or self.sym_tbl.get_symbol(
            name=func_name,
            kind=[
                Symbol.Kind.Arg,
                Symbol.Kind.Var,
                Symbol.Kind.Class,
            ],
        )

        # self.sym_tbl.print_summary()

        if not func:
            self.report_error(node, f"Target function or class '{
                              func_name}' not found")
            return self.verify_FuncCall(node, func)

        elif isinstance(func, ast.VarDecl):
            true_func = Sema.get_true_var(func)
            if isinstance(true_func, ast.LispVarRef):
                if isinstance(true_func, ast.LispVarRef):
                    return self.verify_FuncCall(
                        ast.LispFuncCall(
                            func=true_func, args=node.args, loc=node.loc),
                        func,
                    )
            self.report_error(
                node, f"{node.loc}\nUnknown lisp function call {func}")
            return node

        elif isinstance(func, FuncOverloads):
            the_func = func.lookup(node.args)
            if the_func:
                node = ast.FuncCall(
                    func=the_func, args=node.args, loc=node.loc)
            else:
                self.report_error(node, f"Cannot find a matched function")

            return self.verify_FuncCall(node, the_func)

        else:
            assert func is not None
            node = ast.FuncCall(func=func, args=node.args,
                                loc=node.loc)  # type: ignore
            return self.verify_FuncCall(node, func)

    def verify_FuncCall(
        self,
        node: ast.FuncCall | ast.LispFuncCall,
        func: ast.VarDecl | ast.LispVarRef | None,
    ) -> ast.FuncCall | ast.LispFuncCall:
        if node.sema_failed:
            return node

        if any_failed(*node.args):
            node.sema_failed = True
        if isinstance(node, ast.FuncCall):
            # TODO[superjomn]: check if the number of arguments and type is correct
            return node
        return node

    def visit_LispFuncCall(self, node: ast.LispFuncCall):
        return super().visit_LispFuncCall(node)

    def visit_Block(self, node: ast.Block):
        with self.sym_tbl.scope_guard():
            node = super().visit_Block(node)

            if any_failed(*node.stmts):
                node.sema_failed = True
        return self.verify_Block(node)

    def verify_Block(self, node: ast.Block) -> ast.Block:
        if node.sema_failed:
            return node

        return_types = set()
        # gather return types
        for return_stmt in node.stmts:
            if isinstance(return_stmt, ast.ReturnStmt):
                if return_stmt.value:
                    return_types.add(return_stmt.value.get_type())
        node.return_type = list(return_types) if return_types else [_ty.Nil]
        return node

    def visit_ArgDecl(self, node: ast.ArgDecl):
        node = super().visit_ArgDecl(node)
        assert (
            self.sym_tbl.current_scope.kind is ScopeKind.Func
        ), f"{node.loc}\nArgDecl should be in a function, but get {self.sym_tbl.current_scope.kind}"
        symbol = Symbol(name=node.name, kind=Symbol.Kind.Arg)
        if self.sym_tbl.contains_locally(symbol):
            self.report_error(node, f"Argument {node.name} already exists")
        else:
            node = self.sym_tbl.insert(symbol, node)
        return self.verify_ArgDecl(node)

    def verify_ArgDecl(self, node: ast.ArgDecl) -> ast.ArgDecl:
        if not (node.is_cls_placeholder or node.is_self_placeholder):
            if node.type is _ty.Unk and node.default is None:
                self.report_error(
                    node,
                    f"{node.loc}\nArg {
                        node.name} should have a type or a default value",
                )
            elif node.default is not None and node.type is not _ty.Unk:
                if not can_assign_type(node.type, node.default.get_type()):
                    self.report_error(
                        node,
                        f"Cannot assign {node.default.get_type()} to {
                            node.type}",
                    )
        return node

    def visit_BinaryOp(self, node: ast.BinaryOp):
        node = super().visit_BinaryOp(node)
        return self.verify_BinaryOp(node)

    def verify_BinaryOp(self, node: ast.BinaryOp) -> ast.BinaryOp:
        if any_failed(node.left, node.right):
            node.sema_failed = True
            return node

        op_to_op_check = {
            ast.BinaryOperator.ADD: self.can_type_add,
            ast.BinaryOperator.SUB: self.can_type_sub,
            ast.BinaryOperator.MUL: self.can_type_mul,
            ast.BinaryOperator.DIV: self.can_type_div,
            ast.BinaryOperator.EQ: self.can_type_eq,
            ast.BinaryOperator.NE: self.can_type_neq,
        }

        for op, check in op_to_op_check.items():
            if node.op is op:
                if not check(node.left, node.right):
                    self.report_error(
                        node,
                        f"Cannot {op.name} {node.left.get_type()} and {
                            node.right.get_type()}",
                    )
                    node.sema_failed = True
                    return node

        if not is_type_compatible(node.left.get_type(), node.right.get_type()):
            self.report_error(
                node,
                f"Cannot convert {node.left.get_type()} and {
                    node.right.get_type()}",
            )

        node.type = get_common_type(
            node.left.get_type(), node.right.get_type())
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

    def visit_UnaryOp(self, node: ast.UnaryOp):
        node = super().visit_UnaryOp(node)
        return node

    def verify_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp:
        if node.value.sema_failed:
            node.sema_failed = True
            return node
        return node

    def visit_CallParam(self, node: ast.CallParam):
        node = super().visit_CallParam(node)
        return node

    def verify_CallParam(self, node: ast.CallParam) -> ast.CallParam:
        if node.sema_failed:
            return node
        return node

    @staticmethod
    def get_true_var(
        node: ast.VarDecl | ast.VarRef,
    ) -> ast.VarDecl | ast.LispVarRef | ast.ArgDecl | ast.UnresolvedVarRef | ast.Expr:
        if isinstance(node, ast.VarDecl) and node.init:
            return node.init
        if isinstance(node, ast.VarRef):
            assert node.decl
            return node.decl
        if isinstance(node, ast.LispVarRef):
            return node

        return node
