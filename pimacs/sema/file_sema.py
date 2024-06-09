"""
The FileSema module will scan the whole file and resolve the symbols in the file.
"""
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pimacs.ast.type as _ty
from pimacs.ast.utils import WeakSet
from pimacs.logger import logger
from pimacs.sema.func import FuncOverloads

from . import ast
from .ast import AnalyzedClass, CallMethod, MakeObject, Template, UCallMethod
from .ast_visitor import IRMutator, IRVisitor
from .context import ModuleContext, ScopeKind, Symbol, SymbolTable
from .func import FuncSig
from .type_infer import TypeInfer, is_unk
from .utils import ClassId, FuncSymbol, ModuleId, bcolors, print_colored


def any_failed(*nodes: ast.Node) -> bool:
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


class FileSema(IRMutator):
    """
    Clean up the symbols in the IR, like unify the VarRef or FuncDecl with the same name in the same scope.
    """

    # NOTE, the ir nodes should be created when visited.

    def __init__(self, ctx: ModuleContext):
        self.ctx = ctx
        self._succeeded = True

        self._type_infer = TypeInfer(ctx)

        self._cur_class: ast.Class | None = None
        self._cur_file: ast.File | None = None

        # holds all the unresolved symbol instances
        self._unresolved_symbols: WeakSet = WeakSet()
        # The newly resolved symbols those need to be type-inferred
        self._newly_resolved_symbols: WeakSet = WeakSet()

    def __call__(self, node: ast.Node):
        tree = self.visit(node)
        # First turn of type inference
        self._type_infer(tree)

        # Try to bind the unresolved symbols repeatedly
        remaining = self.bind_unresolved_symbols()
        while remaining > 0 and self.bind_unresolved_symbols() < remaining:
            remaining = self.bind_unresolved_symbols()

        # TODO: the root might be re-bound
        return tree

    def visit(self, node):
        node = super().visit(node)
        self.infer_type()
        return node

    @property
    def sym_tbl(self) -> SymbolTable:
        return self.ctx.symbols

    @property
    def succeed(self) -> bool:
        return self._succeeded

    def report_error(self, node: ast.Node, message: str):
        self.ctx.report_sema_error(node, message)
        node.sema_failed = True
        self._succeeded = False

    def collect_unresolved(self, node: ast.Node):
        logger.debug(f"Collect unresolved symbol {node}")
        assert node.resolved is False, f"{node} is already resolved"
        # bind scope for second-turn name binding
        node.scope = self.sym_tbl.current_scope  # type: ignore
        self._unresolved_symbols.add(node)

    def collect_newly_resolved(self, node: ast.Node):
        # Since the IRMutator cannot get parent in a single visit method, we need to collect the newly resolved symbols and type-infer them in the next turn.
        if not node.sema_failed:
            self._newly_resolved_symbols.add(node)

    def infer_type(self):
        for node in self._newly_resolved_symbols:
            self._type_infer(node, forward_push=False)
        self._newly_resolved_symbols.clear()

    def visit_File(self, node: ast.File):
        self._cur_file = node
        super().visit_File(node)
        return node

    def visit_UVarRef(self, node: ast.UVarRef):
        node = super().visit_UVarRef(node)
        # case 0: self.attr within a class
        if node.name.startswith("self."):
            # deal with members
            var_name = node.name[5:]

            obj = self.sym_tbl.lookup(name="self", kind=Symbol.Kind.Arg)
            if not obj:
                obj = ast.UVarRef(
                    # TODO: fix the type
                    name="self", loc=node.loc, target_type=_ty.Unk)  # type: ignore
                obj.sema_failed = True
                self.report_error(node, f"`self` is not declared")
            else:
                assert self._cur_class
                # TODO: deal with the templated class
                obj.type = _ty.GenericType(self._cur_class.name)

            # The Attr cannot be resolved now, it can only be resolved in bulk when a class is fully analyzed.
            attr = ast.UAttr(value=obj, attr=var_name, loc=node.loc)
            self.collect_unresolved(attr)
            return attr

        # case 1: Lisp symbol
        elif node.name.startswith('%'):
            ret = ast.VarRef(name=node.name, loc=node.loc, type=_ty.LispType)
            self.collect_newly_resolved(ret)
            return ret

        # case 2: a local variable or argument
        elif sym := self.sym_tbl.lookup(
            [Symbol(name=node.name, kind=Symbol.Kind.Var),
             Symbol(name=node.name, kind=Symbol.Kind.Arg)]
        ):
            if isinstance(sym, ast.VarDecl):
                var = ast.VarRef(target=sym, loc=node.loc)  # type: ignore
            elif isinstance(sym, ast.Arg):
                var = ast.VarRef(target=sym, loc=node.loc)
            elif isinstance(sym, ast.VarRef):
                var = sym
            else:
                raise ValueError(f"{node.loc}\nUnknown symbol type {sym}")

            self.collect_newly_resolved(var)
            return var

        else:
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Var)
            self.ctx.report_sema_error(node, f"Symbol {symbol} not found")
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

    def visit_Select(self, node: ast.Select):
        node = super().visit_Select(node)
        if any_failed(node.cond, node.then_expr, node.else_expr):
            node.sema_failed = True
            return node

        return self.verify_SelectExpr(node)

    def verify_SelectExpr(self, node: ast.Select) -> ast.Select:
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

    def func_setup_template(self, node: ast.Function):
        '''
        Parse the template from decorators.
        '''
        if node.template_params is not None:
            return

        for no, decorator in enumerate(node.decorators):
            if isinstance(decorator.action, ast.Template):
                tpl: ast.Template = decorator.action
                # assert all the arg from call.args are Type
                assert all(isinstance(type, _ty.PlaceholderType)
                           for type in tpl.types)
                node.template_params = tpl.types
                break
        if node.template_params is None:  # mark as inited
            node.template_params = tuple()

    def func_pollute_tpl_placeholder_in_func_signature(self, node: ast.Function):
        '''
        Replace the T0,T1 types in the args with the actual PlaceholderTypes.
        This is necessary since the ast parser parse T0,T1 as GenericType and mark them as concrete.
        '''
        if not node.template_params:
            return

        name_to_param = {param.name: param for param in node.template_params}
        args = [arg for arg in node.args]
        for arg in args:
            if param_type := name_to_param.get(arg.type.name, None):
                arg.type = param_type
        node.args = tuple(args)

        if node.return_type:
            if param_type := name_to_param.get(node.return_type.name, None):
                node.return_type = param_type

    '''
    def func_setup_method_self(self, node: ast.Function):
        if self._cur_class:
            if node.args and node.args[0].name == "self":
                args = [arg for arg in node.args]
                args[0].is_self_placeholder = True # type: ignore
                node.args = tuple(args)
    '''

    def visit_Function(self, node: ast.Function):
        self.func_setup_template(node)
        # self.func_setup_method_self(node)
        self.func_pollute_tpl_placeholder_in_func_signature(node)

        within_class = self.sym_tbl.current_scope.kind is ScopeKind.Class
        if within_class:
            assert self._cur_class

        with self.sym_tbl.scope_guard(kind=ScopeKind.Func):
            node = super().visit_Function(node)
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
                    cls_arg.kind = ast.Arg.Kind.cls_placeholder
                elif not node.is_staticmethod:
                    self_arg = args[0]  # self placeholder
                    if self_arg.name != "self":
                        self.report_error(
                            node, f"Method should have the first arg named 'self'"
                        )
                    self_arg.kind = ast.Arg.Kind.self_placeholder

            if any_failed(*node.args, node.body, *node.decorators):
                node.sema_failed = True

        node = self.sym_tbl.insert(symbol, node)
        return self.verify_FuncDecl(node)

    def verify_FuncDecl(self, node: ast.Function) -> ast.Function:
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
            if arg.name == "self" and arg.kind is ast.Arg.Kind.self_placeholder:
                pass
            elif arg.name == "cls" and arg.kind is ast.Arg.Kind.cls_placeholder:
                pass
            elif not arg.type or arg.type is _ty.Unk:
                self.report_error(
                    arg, f"Argument {arg.name} should have a type")
        # check if return type is valid
        if not node.return_type or node.return_type is _ty.Unk:
            node.return_type = _ty.Void
            return_nil = (not node.body.return_type) or (
                len(
                    node.body.return_type) == 1 and node.body.return_type[0] is _ty.Void
            )
            if not return_nil:
                self.report_error(
                    node,
                    f"Function which returns non-nil values should set return type",
                )
        elif node.body.return_type and node.body.return_type is not _ty.Void:
            if not can_assign_type(node.return_type, node.body.return_type):
                self.report_error(
                    node,
                    f"Cannot convert {node.body.return_type} to {
                        node.return_type}",
                )

        return node

    def visit_Class(self, the_node: ast.Class):
        node = AnalyzedClass.create(the_node)  # type: ignore
        with node.auto_update_symbols():
            if self.sym_tbl.current_scope.kind is not ScopeKind.Global:
                self.report_error(
                    node, f"Class should be declared in the global scope")

            with self.sym_tbl.scope_guard(kind=ScopeKind.Class):
                with self.class_scope_guard(node):
                    node = super().visit_Class(node)

            symbol = Symbol(name=node.name, kind=Symbol.Kind.Class)
            if self.sym_tbl.contains_locally(symbol):
                self.report_error(node, f"Class {node.name} already exists")

            node = self.sym_tbl.insert(symbol, node)

        # Add constructors to the class into the global scope
        self._add_class_constructors(node)

        return self.verify_Class(node)

    def _add_class_constructors(self, node: AnalyzedClass):
        ''' Add constructor functions to the class. '''
        symbol = FuncSymbol("__init__")
        overloads = node.symbols.get(symbol)

        assert self._cur_file
        if overloads is None:
            # add a default constructor
            fn = self._get_class_default_constructor(node)

            # append to the file for sema later
            self._cur_file.stmts.append(fn)
        else:
            # add constructor for each __init__ method
            for init_fn in overloads:
                fn = self._get_class_constructor(node.name, init_fn)
                symbol = FuncSymbol(fn.name)
                if overloads := self.sym_tbl.global_scope.get(symbol):
                    if overloads.lookup(FuncSig.create(fn)):
                        self.report_error(
                            node, f"Constructor {fn.name} already exists")

                # append to the file for sema later
                self._cur_file.stmts.append(fn)

    def _add_class_methods(self, node: AnalyzedClass):
        ''' Add method functions to the class. '''
        members = node.symbols.get(Symbol.Kind.Member)
        assert self._cur_file
        for member in members:
            if isinstance(member, ast.Function) and member.name != "__init__":
                method = self._add_method(node.name, member)
                symbol = FuncSymbol(
                    method.name, annotation=ast.Function.Annotation.Class_method)
                if overloads := self.sym_tbl.global_scope.get(symbol):
                    if overloads.lookup(FuncSig.create(method)):
                        self.report_error(
                            node, f"Class method {method.name} duplicates")

                # append to the file for sema later
                self._cur_file.stmts.append(method)

    def _get_class_default_constructor(self, node: AnalyzedClass) -> ast.Function:
        '''
        Create a default constructor for the class.
        '''
        members = node.symbols.get(Symbol.Kind.Member)
        args = []
        for member in members:
            arg = ast.Arg(name=member.name, loc=member.loc)
            arg.type = member.type
            if member.init:
                arg.default = member.init
            args.append(arg)

        return_type = _ty.GenericType(node.name)

        make_obj_expr = MakeObject(class_name=node.name, loc=node.loc)
        make_obj_expr.type = return_type

        return_stmt = ast.Return(value=make_obj_expr, loc=node.loc)

        body = ast.Block(stmts=(return_stmt,
                                ), loc=node.loc)
        fn = ast.Function(name=node.name, args=tuple(args),
                          body=body, loc=node.loc, return_type=return_type)
        fn.annotation = ast.Function.Annotation.Class_constructor
        return fn

    def _get_class_constructor(self, class_name: str, init_fn: ast.Function):
        '''
        Create a constructor function for the class.
        '''
        assert init_fn.args[0].name == "self"
        args = init_fn.args[1:]
        body = init_fn.body
        return_type = init_fn.args[0].type
        fn = ast.Function(name=class_name, args=tuple(args), body=body, loc=init_fn.loc,
                          return_type=return_type)
        fn.annotation = ast.Function.Annotation.Class_constructor
        return fn

    def _add_method(self, class_name: str, method: ast.Function):
        '''
        Create a method function for the class in the global scope.
        The method will guard with annotation `Class_method`, and the first argument should be `self`.
        '''
        assert method.args[0].name == "self"
        args = method.args
        body = method.body
        return_type = method.return_type
        fn = ast.Function(name=method.name, args=tuple(args), body=body, loc=method.loc,
                          return_type=return_type)
        fn.annotation = ast.Function.Annotation.Class_method
        return fn

    def verify_Class(self, node: AnalyzedClass) -> ast.Class:
        if node.sema_failed:
            return node
        return node

    def visit_Call(self, node: ast.Call):
        node = super().visit_Call(node)

        func = node.func

        match type(func):
            case ast.UFunction:
                func_name = func.name  # type: ignore
                func_overloads = self.sym_tbl.lookup(FuncSymbol(func_name))
                if func_overloads and (func_candidates := func_overloads.lookup(node.args)):
                    if len(func_candidates) > 1:
                        self.report_error(
                            node, f"Function {func_name} has more than one candidates")
                        node.sema_failed = True
                    elif len(func_candidates) == 1:
                        logger.debug(f"Resolved function {
                                     func_name}: {func_candidates}")
                        func, func_sig = func_candidates[0]
                        node.func = func  # bind the func
                        node.type = func_sig.output_type
                        self.collect_newly_resolved(node)
                else:
                    self.collect_unresolved(node.func)  # type: ignore

            case ast.UAttr:
                new_node = UCallMethod(
                    obj=func.value, attr=func.attr,  # type: ignore
                    args=node.args, loc=func.loc)  # type: ignore
                logger.debug(f"ast.UAttr users: {node.users}")
                node.replace_all_uses_with(new_node)
                logger.debug(f"ast.UCallAttr users: {new_node.users}")
                self.collect_unresolved(new_node)
                return new_node

            case _:
                assert isinstance(func, str), f"call: {node}"

        # TODO: check if the function is a member function
        # TODO: check if the function is a class

        return self.verify_Call(node, node.func)  # type: ignore

    def verify_Call(
        self,
        node: ast.Call,
        func: ast.VarDecl | ast.Function | None,
    ) -> ast.Call:
        if node.sema_failed:
            return node

        if any_failed(*node.args):
            node.sema_failed = True
        if isinstance(node, ast.Call):
            # TODO[superjomn]: check if the number of arguments and type is correct
            return node
        return node

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
            if isinstance(return_stmt, ast.Return):
                if return_stmt.value:
                    return_types.add(return_stmt.value.get_type())
        node.return_type = list(return_types) if return_types else [_ty.Void]
        return node

    @contextmanager
    def class_scope_guard(self, class_node: ast.Class):
        self._cur_class = class_node
        yield
        self._cur_class = None

    def visit_Arg(self, node: ast.Arg):
        node = super().visit_Arg(node)
        assert (
            self.sym_tbl.current_scope.kind is ScopeKind.Func
        ), f"{node.loc}\nArgDecl should be in a function, but get {self.sym_tbl.current_scope.kind}"

        if self._cur_class:
            if node.name == "self":
                node.kind = ast.Arg.Kind.self_placeholder
                if not is_unk(node.type):
                    self.report_error(node, f"self should not have a type")
            elif node.name == "cls":
                node.kind = ast.Arg.Kind.cls_placeholder
                if not is_unk(node.type):
                    self.report_error(node, f"self should not have a type")

        symbol = Symbol(name=node.name, kind=Symbol.Kind.Arg)
        if self.sym_tbl.contains_locally(symbol):
            self.report_error(node, f"Argument {node.name} already exists")
        else:
            node = self.sym_tbl.insert(symbol, node)
        return self.verify_Arg(node)

    def verify_Arg(self, node: ast.Arg) -> ast.Arg:
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
        elif node.is_self_placeholder:
            assert self._cur_class
            if not is_unk(node.type):
                self.report_error(
                    node,
                    f"{node.loc}\nself should not have a type",
                )
            else:
                node.type = self._cur_class.as_type()
        return node

    def visit_BinaryOp(self, node: ast.BinaryOp):
        node = super().visit_BinaryOp(node)
        return self.verify_BinaryOp(node)

    def verify_BinaryOp(self, node: ast.BinaryOp) -> ast.BinaryOp:
        if any_failed(node.left, node.right):
            node.sema_failed = True
            return node
        if not (node.left.type_determined and node.right.type_determined):
            # Not all the operands's type are determined
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

    def is_type_numeric(self, type: _ty.Type) -> bool:
        return type in (_ty.Int, _ty.Float)

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
        if node.operand.sema_failed:
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

    def visit_MakeObject(self, node: MakeObject):
        return node

    def visit_UAttr(self, node: ast.UAttr):
        node = super().visit_UAttr(node)
        self.collect_unresolved(node)
        return node

    @staticmethod
    def get_true_var(
        node: ast.VarDecl | ast.VarRef,
    ) -> ast.VarDecl | ast.Arg | ast.UVarRef | ast.Expr:
        if isinstance(node, ast.VarDecl) and node.init:
            return node.init
        if isinstance(node, ast.VarRef):
            assert node.target
            return node.target

        return node

    def bind_unresolved_symbols(self) -> int:
        '''
        Try to bind the unresolved symbols.
        Return the number of resolved symbols.
        '''
        logger.debug(f"unresolved symbols: {self._unresolved_symbols}")
        resolved = []
        for node in self._unresolved_symbols:
            if self.bind_unresolved(node):
                resolved.append(node)
        for node in resolved:
            self._unresolved_symbols.remove(node)

        return len(resolved)

    utypes = ast.UVarRef | ast.UAttr | ast.UFunction | ast.UClass

    def bind_unresolved(self, node: utypes) -> bool:
        '''
        Try to bind the unresolved symbol.

        Return True if the symbol is resolved.
        '''
        logger.debug(f"Try to bind unresolved symbol {node}")
        if node.resolved:
            return True

        match type(node):
            case ast.UFunction:
                assert isinstance(node, ast.UFunction)  # pass the typing
                logger.debug(f"Bind unresolved function {node}")
                func_overloads = node.scope.get(
                    FuncSymbol(node.name))  # type: ignore
                logger.debug(f"resolving UFunction found overloads: {
                             func_overloads}")
                if func_overloads is None:
                    return False  # remain unresolved
                assert len(node.users) == 1  # only one caller
                call = list(node.users)[0]
                assert isinstance(call, ast.Call)
                # assert func_overloads.lookup(call.args)

                if func_candidates := func_overloads.lookup(call.args):
                    if len(func_candidates) > 1:
                        self.report_error(
                            node, f"Function {node.name} has more than one candidates")
                        return False
                    elif not func_candidates:
                        return False

                    func, func_sig = func_candidates[0]

                    node.replace_all_uses_with(func)
                    self._type_infer(func, forward_push=False)
                    return True
                return False

            case ast.UVarRef:
                assert not isinstance(node, ast.UAttr)
                symbol = Symbol(name=node.name, kind=Symbol.Kind.Var)

                if sym := node.scope.lookup(symbol):
                    node.replace_all_uses_with(sym)

                    self._type_infer(sym, forward_push=False)
                    return True
                return False

            case ast.UAttr:
                attr_name = node.attr  # type: ignore
                if node.value.get_type() in (None, _ty.Unk):  # type: ignore
                    return False

                class_symbol = Symbol(
                    name=node.value.get_type().name, kind=Symbol.Kind.Class)  # type: ignore
                class_node = self.sym_tbl.global_scope.get(class_symbol)

                member = class_node.symbols.get(
                    Symbol(name=attr_name, kind=Symbol.Kind.Member))
                if not member:
                    self.report_error(node, f"Attribute {attr_name} not found")
                    return False

                new_node = ast.Attribute(
                    value=node.value, attr=attr_name, loc=node.loc)  # type: ignore
                new_node.type = member.type
                node.replace_all_uses_with(new_node)

                self._type_infer(new_node, forward_push=False)
                return True

            case ast.UCallMethod:
                if node.obj.get_type() in (None, _ty.Unk):
                    logger.debug(f"Cannot resolve {node.obj}")
                    return False

                class_symbol = Symbol(
                    name=node.obj.get_type().name, kind=Symbol.Kind.Class)
                class_node = self.sym_tbl.global_scope.get(class_symbol)
                assert class_node

                func_symbol = FuncSymbol(node.attr)
                methods: Optional[FuncOverloads] = class_node.symbols.get(
                    func_symbol)
                if not methods:
                    self.report_error(node, f"No method called {
                                      node.attr} in class {node.obj.get_type()}")
                    return False

                args = tuple([node.obj] + list(node.args))  # self + args
                if method_candidates := methods.lookup(args):
                    logger.debug(f"UCallAttr found method: {
                                 method_candidates}")
                    if len(method_candidates) > 1:
                        self.report_error(
                            node, f"Method {node.attr} has more than one candidates")
                        return False
                    elif not method_candidates:
                        return False

                    method, method_sig = method_candidates[0]

                    new_node = CallMethod(
                        obj=node.obj, method=method, args=node.args, loc=node.loc)
                    new_node.type = method.return_type

                    node.replace_all_uses_with(new_node)
                    self._type_infer(new_node, forward_push=False)
                    self.collect_newly_resolved(new_node)
                    return True
                else:
                    self.report_error(node, f"Method {node.attr} not found")
                    return False

            case ast.UClass:
                # TODO: resolve the class
                logger.debug(f"TODO Bind unresolved class {node}")
                return False

        return False
