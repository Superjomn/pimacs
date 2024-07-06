'''
The semantics analysis for Class.
'''
from contextlib import contextmanager
from typing import Any

import pimacs.ast.type as _ty
from pimacs.logger import get_logger

from . import ast
from .ast import AnalyzedClass, MakeObject
from .ast_visitor import IRMutator
from .func import FuncSig
from .symbol_table import FuncSymbol, ScopeKind, Symbol, SymbolTable

logger = get_logger(__name__)


class ClassVisitor(IRMutator):
    def __init__(self, file_sema: Any) -> None:
        self.file_sema = file_sema

    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        if method := self.__dict__.get(method_name):
            return getattr(self, method_name)(node)
        return getattr(self.file_sema, method_name)(node)

    @contextmanager
    def class_scope_guard(self, class_node: ast.Class):
        self.file_sema._cur_class = class_node
        yield
        self.file_sema._cur_class = None

    @property
    def sym_tbl(self) -> SymbolTable:
        return self.file_sema.ctx.symbols

    @property
    def _cur_file(self):
        return self.file_sema._cur_file

    @property
    def succeed(self) -> bool:
        return self.file_sema._succeeded

    def report_error(self, node: ast.Node, message: str):
        self.file_sema.ctx.report_sema_error(node, message)
        node.sema_failed = True
        self.file_sema._succeeded = False

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
        self.class_add_constructors(node)

        return self.verify_Class(node)

    def class_add_constructors(self, node: AnalyzedClass):
        ''' Add constructor functions to the class.
        This will create individual functions in global scope to help create instances of the class.

        e.g.

        class App:
            def __init__(a: Int):
                return

        will get a function:

        def App(a: Int) -> App:
            # create instance
            # same body of App.__init__(a)
            return instance
        '''
        symbol = FuncSymbol("__init__")
        overloads = node.symbols.get(symbol)

        assert self._cur_file
        if overloads is None:
            # add a default constructor
            fn = self.class_create_default_constructor(node)

            # append to the file for sema later
            self._cur_file.stmts.append(fn)
        else:
            # add constructor for each __init__ method
            for init_fn in overloads:
                fn = self.class_create_constructor(node, init_fn)
                symbol = FuncSymbol(fn.name)
                if overloads := self.sym_tbl.global_scope.get(symbol):
                    if overloads.lookup(FuncSig.create(fn)):
                        self.report_error(
                            node, f"Constructor {fn.name} already exists")

                # append to the file for sema later
                self._cur_file.stmts.append(fn)
                logger.debug(f"adding constructor {fn.name} to cur_file")

    def class_add_methods(self, node: AnalyzedClass):
        ''' Add method functions to the class. '''
        members = node.symbols.get(Symbol.Kind.Member)
        assert self._cur_file
        for member in members:
            if isinstance(member, ast.Function) and member.name != "__init__":
                method = self.class_add_method(node.name, member)
                symbol = FuncSymbol(
                    method.name, annotation=ast.Function.Annotation.Class_method)
                if overloads := self.sym_tbl.global_scope.get(symbol):
                    if overloads.lookup(FuncSig.create(method)):
                        self.report_error(
                            node, f"Class method {method.name} duplicates")

                # append to the file for sema later
                self._cur_file.stmts.append(method)

    def class_create_default_constructor(self, node: AnalyzedClass) -> ast.Function:
        '''
        Create a default constructor for the class.
        '''
        # TODO: keep the order of members in the arg-list
        members = node.symbols.get(Symbol.Kind.Member)

        # create the arg-list
        args = []
        for member in members:
            arg = ast.Arg(name=member.name, loc=member.loc)
            arg.type = member.type
            if member.init:
                arg.default = member.init
            args.append(arg)

        if node.template_params:
            return_type = _ty.CompositeType(node.name, node.template_params)
        else:
            return_type = _ty.GenericType(node.name)  # type: ignore

        # TODO: create the body of the constructor
        make_obj_expr = MakeObject(loc=node.loc)
        make_obj_expr.type = return_type

        return_stmt = ast.Return(value=make_obj_expr, loc=node.loc)

        body = ast.Block(stmts=(return_stmt,
                                ), loc=node.loc)
        fn = ast.Function(name=node.name, args=tuple(args),
                          template_params=node.template_params,
                          body=body, loc=node.loc, return_type=return_type)

        fn.annotation = ast.Function.Annotation.Class_constructor
        return fn

    def class_create_constructor(self, class_node: AnalyzedClass, init_fn: ast.Function):
        '''
        Create a constructor function for the class.
        '''
        assert init_fn.args[0].name == "self"
        args = init_fn.args[1:]
        body = init_fn.body
        return_type = _ty.CompositeType(
            class_node.name, params=class_node.template_params)

        # parepare the body
        # Insert `self = make_object()`
        obj = MakeObject(loc=init_fn.loc)
        obj.type = return_type
        var = ast.VarDecl(name="self", loc=init_fn.loc, init=obj)
        var_ref = ast.VarRef(target=var, loc=init_fn.loc)
        return_stmt = ast.Return(value=var_ref, loc=init_fn.loc)
        body.stmts = (var,) + body.stmts + (return_stmt,)

        logger.debug(f"create constructor {class_node.name} with self {obj}")

        fn = ast.Function(name=class_node.name, args=tuple(args), body=body, loc=init_fn.loc,
                          template_params=class_node.template_params,
                          return_type=return_type)
        fn.annotation = ast.Function.Annotation.Class_constructor

        return fn

    def class_add_method(self, class_name: str, method: ast.Function):
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

        self.file_sema.verify_template(node.template_params, node)

        return node
