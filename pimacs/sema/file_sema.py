"""
The FileSema module will scan the whole file and resolve the symbols in the file.
"""
from contextlib import contextmanager
from typing import List, Tuple

import pimacs.ast.type as _ty
from pimacs.ast.utils import WeakSet
from pimacs.logger import get_logger

from . import ast
from .ast import MakeObject, UCallMethod
from .ast_visitor import IRMutator
from .class_sema import ClassVisitor
from .context import ModuleContext, Symbol, SymbolTable
from .name_binder import NameBinder
from .type_checker import TypeChecker, is_unk
from .utils import FuncSymbol, ScopeKind

logger = get_logger(__name__)


def any_failed(*nodes: ast.Node) -> bool:
    return any(node.sema_failed for node in nodes)


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
    Performing Sema on a single File which is actually a module.
    During Sema, it will resolve the symbols in the module, and type-infer the types of the nodes.

    The algorithm for resolving symbols(NameBinder) is as follows:
    1. Doing a bottom-up traversal of the AST
        a. It will insert the resolved sybmols into the symbol table, such as VarDecl, Function, Class, etc.
        b. It will collect the unresolved symbols such as UVarRef, UAttr, UFunction, UClass, etc, ideally these symbols
              should be resolved in the subsequent turn of name binding.
    2. For the unresolved symbols, it will try to bind them eagerly until no more symbols are resolved. This is done by
        the `NameBinder`.

    The algorithm for type-check(TypeChecker) is as follows:
    1. For any newly resolved symbols, it will try to type-infer them.
        a. It will first try to infer the type in an forward-push way, which means it will try to infer the children's
            types. For instance, a VarDecl(init=Literal(1)), it will start from the VarDecl node, and recursively arrive
            on the Literal node and update its type to Int.
        b. Once any node's type is inferred, it will try to infer the parent's type in a backward-push way. For instance,
            a VarDecl(init=Literal(1)), after the Literal's type is inferred, it will try to infer the VarDecl's type.

    These two algorithms will be triggered in a chain, for example, once a symbol is newly resolved, it will trigger the
    type-checking, and got a valid type, and all its users' types could be inferred. And once a node's type is inferred,
    it could help name binding, for instance, a UAttr(node=self, attr="name"), once the type of self is inferred to some
    Class, the attr could be resolved to a member of the Class.
    """

    # NOTE, the ir nodes should be created when visited.

    def __init__(self, ctx: ModuleContext):
        self.ctx = ctx
        self._succeeded = True

        self.type_checker = TypeChecker(ctx)
        self.name_binder = NameBinder(self, self.type_checker)

        self._cur_class: ast.AnalyzedClass | None = None
        self._cur_func: ast.Function | None = None
        self._cur_file: ast.File | None = None

        # holds all the unresolved symbol instances
        self._unresolved_symbols: WeakSet = WeakSet()
        # The newly resolved symbols those need to be type-inferred
        self._newly_resolved_symbols: WeakSet = WeakSet()

        self._class_visitor = ClassVisitor(self)

        self._cur_module = ast.Module(name=ctx.name, loc=None, path=None)
        self._cur_module.ctx = ctx

    def __call__(self, node: ast.Node):
        tree = self.visit(node)
        # First turn of type inference
        self.type_checker(tree)

        self._resolve_symbols_eagarly()

        # TODO: the root might be re-bound
        return tree

    def _resolve_symbols_eagarly(self):
        # Try to bind the unresolved symbols repeatedly until no more symbols are resolved
        remaining = self.bind_unresolved_symbols()
        while remaining > 0 and self.bind_unresolved_symbols() < remaining:
            remaining = self.bind_unresolved_symbols()
            # TODO: Unify the check_type timing
            self.check_type()

    def visit(self, node):
        node = super().visit(node)
        self.check_type()
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
        '''
        Collect the unresolved symbol node for the second turn of name binding after the first AST traversal is done.
        '''
        logger.debug(f"Collect unresolved symbol {node}")
        assert node.resolved is False, f"{node} is already resolved"
        # bind scope for second-turn name binding
        node.scope = self.sym_tbl.current_scope  # type: ignore
        self._unresolved_symbols.add(node)

    def collect_newly_resolved(self, node: ast.Node):
        '''
        Collect the newly resolved symbol node for the second turn of type inference.
        '''
        # Since the IRMutator cannot get parent in a single visit method, we need to collect the newly resolved
        # symbols and type-infer them in the next turn. The type-infer will trigger in chain.
        if (not node.sema_failed) and node.resolved:
            self._newly_resolved_symbols.add(node)

    def link_modules(self, modules: List[ast.Module]):
        self.name_binder.name_to_module = {
            module.name: module for module in modules}

        # trigger the second turn of name binding
        self._resolve_symbols_eagarly()

    def check_type(self):
        finished_nodes = []
        for node in self._newly_resolved_symbols:
            if is_unk(node.type):
                self.type_checker(node, forward_push=True)
            if not is_unk(node.type):
                finished_nodes.append(node)
                # bottom-up pollute
                self.type_checker(node, forward_push=False)
        for node in finished_nodes:
            self._newly_resolved_symbols.remove(node)

    def visit_File(self, node: ast.File):
        self._cur_file = node
        super().visit_File(node)
        return node

    def visit_ImportDecl(self, node: ast.ImportDecl):
        # verify the import statement
        if self.sym_tbl.current_scope.kind is not ScopeKind.Global:
            self.report_error(
                node, f"import statement should be in the global scope")

        if node.module == self.ctx.name:
            self.report_error(node, f"Cannot import the current module itself")

        if node.symbols:
            if len(node.symbols) != len(set(node.symbols)):
                self.report_error(
                    node, f"Duplicate symbols in import statement")

        # insert new symbols
        module_node = ast.UModule(name=node.module, loc=node.loc)
        if not node.symbols:
            # `import a-module` or `import a-module as xx`
            alias = node.alias if node.alias else node.module
            self.sym_tbl.insert(
                Symbol(name=alias, kind=Symbol.Kind.Module), module_node)

        elif len(node.symbols) == 1:
            # `from a-module import a-symbol` or `from a-module import a-symbol as xx`
            alias = node.alias if node.alias else node.symbols[0]
            module_node = ast.UModule(name=node.module, loc=node.loc)
            symbol_node = ast.UAttr(value=module_node,
                                    attr=node.symbols[0], loc=node.loc)

            self.sym_tbl.insert(
                Symbol(name=alias, kind=Symbol.Kind.Unk), symbol_node)
            self.collect_unresolved(symbol_node)

        else:
            # `from a-module import a-symbol1, a-symbol2`
            for symbol in node.symbols:
                symbol_node = ast.UAttr(
                    value=module_node, attr=symbol, loc=node.loc)
                self.sym_tbl.insert(
                    Symbol(name=symbol, kind=Symbol.Kind.Unk), symbol_node)

                self.collect_unresolved(symbol_node)

        self.collect_unresolved(module_node)

    def visit_UVarRef(self, node: ast.UVarRef):
        # case 0: self.attr within a class
        if node.name == "self":
            obj = self.sym_tbl.lookup(
                [Symbol(name="self", kind=Symbol.Kind.Arg),
                 Symbol(name="self", kind=Symbol.Kind.Var)])

            if not obj:
                self.sym_tbl.print_summary()
                obj = ast.UVarRef(
                    # TODO: fix the type
                    name="self", loc=node.loc, target_type=_ty.Unk)  # type: ignore
                self.report_error(node, f"`self` is not declared")
            else:
                if is_unk(obj.type):
                    assert self._cur_class
                    # TODO: deal with the templated class
                    obj.type = self._cur_class.as_type()

            node.replace_all_uses_with(obj)
            return obj

        # case 1: Lisp symbol
        elif node.name.startswith('%'):
            ret = ast.VarRef(name=node.name, loc=node.loc, type=_ty.LispType)
            self.collect_newly_resolved(ret)
            return ret

        # case 2: a local variable or argument
        elif sym := self.sym_tbl.lookup(
            [Symbol(name=node.name, kind=Symbol.Kind.Var),
             Symbol(name=node.name, kind=Symbol.Kind.Arg),
             Symbol(name=node.name, kind=Symbol.Kind.Module),  # for import
             Symbol(name=node.name, kind=Symbol.Kind.Unk),  # for import alias
             ]
        ):
            if isinstance(sym, ast.VarDecl):
                var = ast.VarRef(target=sym, loc=node.loc)  # type: ignore
            elif isinstance(sym, ast.Arg):
                var = ast.VarRef(target=sym, loc=node.loc)
            elif isinstance(sym, ast.VarRef):
                var = sym
            elif isinstance(sym, ast.UModule):
                var = sym  # type: ignore
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
                if not self.type_checker.convert_type(node.init.get_type(), node.type):
                    if is_unk(node.init.get_type()):
                        logger.warning(
                            f"assigning {node.init.get_type()} to {node.type}")
                    else:
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

    def visit_Function(self, node: ast.Function):
        with self.cur_func_guard(node):
            within_class = self.sym_tbl.current_scope.kind is ScopeKind.Class

            if within_class:
                assert self._cur_class

            with self.sym_tbl.scope_guard(kind=ScopeKind.Func):
                node = super().visit_Function(node)
                symbol = FuncSymbol(node.name)

                # TODO[Superjomn]: support function overloading
                if self.sym_tbl.contains_locally(symbol):
                    self.report_error(node, f"Function {
                                      node.name} already exists")

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
            return self.verify_Function(node)

    def verify_Function(self, node: ast.Function) -> ast.Function:
        if node.sema_failed:
            return node

        if node.template_params is not None:
            self.verify_template(node.template_params, node)

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
            if not self.type_checker.convert_type(node.body.return_type, node.return_type):
                self.report_error(
                    node,
                    f"Cannot convert {node.body.return_type} to {
                        node.return_type}",
                )

        return node

    def verify_template(self, params: Tuple[_ty.Type, ...], node: ast.Node):
        if not params:
            return
        # check placeholder unique
        type_set = set(params)
        if len(type_set) != len(params):
            self.report_error(
                node, f"Template params should be unique, got {params}")

    def visit_Class(self, the_node: ast.Class):
        return self._class_visitor.visit_Class(the_node)

    def visit_Call(self, node: ast.Call):
        node.args = self.visit(node.args)
        # node.type_spec = tuple([self.visit(_) for _ in node.type_spec])

        func = node.target
        match type(func):
            case ast.UFunction:  # call a global function
                # Check if the function is imported from another module, and replace the Call with a UCallMethod
                # The UCallMethod has a support for both Class.method and Module.function
                assert hasattr(func, "name")
                if sym := self.sym_tbl.lookup(func.name, [Symbol.Kind.Unk]):
                    if isinstance(sym, ast.UAttr):
                        assert isinstance(sym.value, ast.UModule)
                        new_node = ast.UCallMethod(obj=sym.value, attr=sym.attr, args=node.args,  # type: ignore
                                                   loc=node.loc, type_spec=node.type_spec)
                        node.replace_all_uses_with(new_node)
                        self.collect_unresolved(new_node)
                        return new_node
                    else:
                        raise NotImplementedError(f"call: {node}")
                else:
                    # normal function call
                    self.collect_unresolved(node.target)  # type: ignore

            case ast.UAttr:
                base = self.visit(func.value)  # type: ignore
                new_node = UCallMethod(
                    obj=base, attr=func.attr,  # type: ignore
                    args=node.args, loc=func.loc,  # type: ignore
                    type_spec=node.type_spec)
                node.replace_all_uses_with(new_node)
                self.visit(new_node)
                self.collect_unresolved(new_node)
                return new_node

            case _:
                raise NotImplementedError(f"call: {node}")

        return self.verify_Call(node, node.target)  # type: ignore

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
        node.return_type = list(return_types) if return_types else [_ty.Nil]
        return node

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
                if not self.type_checker.convert_type(node.default.get_type(), node.type):
                    if is_unk(node.default.get_type()):
                        logger.warning(
                            f"assigning {node.default.get_type()} to {node.type}")
                    else:
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

        def check_type_binary(left: _ty.Type, right: _ty.Type):
            if left.is_concrete and right.is_concrete:
                return self.type_checker.convert_type(left, right)

            parent_node = self._cur_func or self._cur_class
            assert parent_node

            assert node.loc
            if left.is_concrete or right.is_concrete:
                # one of them is concrete
                con_ty = left if left.is_concrete else right
                tpl_ty = right if left.is_concrete else left
                assert isinstance(tpl_ty, _ty.PlaceholderType)

                return self.type_checker.convert_to_template_param_type(parent_node=parent_node,
                                                                        source=con_ty, target=tpl_ty, loc=node.loc)

            else:
                # both are template params
                assert isinstance(right, _ty.PlaceholderType)
                return self.type_checker.convert_to_template_param_type(parent_node=parent_node,
                                                                        source=left, target=right, loc=node.loc)

        assert node.left.type and node.right.type
        target_type = check_type_binary(node.left.type, node.right.type)
        if not target_type:
            self.report_error(
                node,
                f"Cannot convert {node.left.type} and {node.right.type}",
            )
            return node

        node.type = target_type
        return node

    def can_type_add(self, left, right):
        return self.type_checker.is_type_numeric(left.get_type()) and self.type_checker.is_type_numeric(
            right.get_type()
        )

    def can_type_sub(self, left, right):
        return self.type_checker.is_type_numeric(left.get_type()) and self.type_checker.is_type_numeric(
            right.get_type()
        )

    def can_type_mul(self, left, right):
        return self.type_checker.is_type_numeric(left.get_type()) and self.type_checker.is_type_numeric(
            right.get_type()
        )

    def can_type_div(self, left, right):
        return self.type_checker.is_type_numeric(left.get_type()) and self.type_checker.is_type_numeric(
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
        if isinstance(node.value, ast.UVarRef):
            node.value = self.visit_UVarRef(node.value)
            if node.value is None:
                self.sym_tbl.print_summary()
                self.report_error(node, f"Cannot resolve {node.value}")
        elif isinstance(node.value, (ast.VarRef, ast.VarDecl, ast.Arg)):
            pass
        else:
            raise NotImplementedError(
                f"UAttr: {node}, type: {type(node.value)}")

        if not node.value.resolved:
            self.sym_tbl.print_summary()
            self_arg = self.sym_tbl.lookup(
                Symbol(name=node.value.name, kind=Symbol.Kind.Arg))
            self.report_error(
                node, f"UAttr: {node} value is not resolved, got {self_arg}")

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
            return node.target  # type: ignore

        return node

    def bind_unresolved_symbols(self) -> int:
        '''
        Try to bind the unresolved symbols.

        Returns:
            the number of resolved symbols.
        '''
        logger.debug(f"unresolved symbols: {self._unresolved_symbols}")
        resolved = []
        for node in self._unresolved_symbols:
            if self.bind_unresolved(node) or not node.users:
                resolved.append(node)
                self.collect_newly_resolved(node)
        for node in resolved:
            self._unresolved_symbols.remove(node)

        return len(resolved)

    utypes = ast.UVarRef | ast.UAttr | ast.UFunction | ast.UClass

    def bind_unresolved(self, node: ast.Node) -> bool:
        if node.resolved:
            return True
        return self.name_binder(node)

    @contextmanager
    def cur_func_guard(self, func: ast.Function):
        self._cur_func = func
        yield
        self._cur_func = None
