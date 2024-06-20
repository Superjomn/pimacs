from typing import Any, Optional

import pimacs.ast.type as _ty
from pimacs.logger import logger

from . import ast
from .func import FuncOverloads
from .utils import FuncSymbol, Symbol


class NameBinder:
    '''
    NameBinder is a helper class to bind unresolved names in the AST.
    '''

    def __init__(self, file_sema: Any, type_checker: Any):
        self.type_checker = type_checker
        self.file_sema = file_sema

    def report_error(self, node: ast.Node, message: str):
        self.file_sema.report_error(node, message)

    @property
    def sym_tbl(self):
        return self.file_sema.sym_tbl

    def __call__(self, node: ast.Node) -> bool:
        assert node.resolved is False, f"Node {node} is already resolved"
        return self.bind_unresolved(node)

    def bind_unresolved(self, node: ast.Node) -> bool:
        if node.resolved:
            return True

        method_name = f'visit_{type(node).__name__}'
        if method := self.__dict__.get(method_name):
            return getattr(self, method_name)(node)
        return getattr(self, method_name)(node)

    def visit_UFunction(self, node) -> bool:
        assert isinstance(node, ast.UFunction)  # pass the typing
        logger.debug(f"Bind unresolved function {node}")

        func_overloads = node.scope.get(
            FuncSymbol(node.name))  # type: ignore
        logger.debug(f"resolving UFunction {node} found overloads: {
            func_overloads}")
        if func_overloads is None:
            logger.debug(f"Cannot resolve {node}")
            return False  # remain unresolved

        assert len(node.users) == 1  # only one caller
        call = list(node.users)[0]
        assert isinstance(call, ast.Call)

        if func_candidates := func_overloads.lookup(call.args, template_spec=call.type_spec):
            logger.debug(f"UCall found function: {func_candidates}")
            if len(func_candidates) > 1:
                self.report_error(
                    node, f"Function {node.name} has more than one candidates")
                return False
            elif not func_candidates:
                return False

            func, func_sig = func_candidates[0]
            logger.debug(f"bind_unresolved: Resolved sig {
                func_sig}, func: {func}")
            if not func_sig.all_param_types_concrete():
                return False

            node.replace_all_uses_with(func)
            if func.template_params:
                # The templated func's output could be a non-concrete type
                call.type = func_sig.output_type
                self.type_checker(call, forward_push=False)
            else:
                self.type_checker(func, forward_push=False)
            return True

        return False

    def visit_UVarRef(self, node: ast.UVarRef) -> bool:
        # Ideally, the UVarRef should be bound during the normal file sema. If failed, it will do global binding here.
        assert not isinstance(node, ast.UAttr)
        symbol = Symbol(name=node.name, kind=Symbol.Kind.Var)

        if sym := node.scope.get(symbol):
            node.replace_all_uses_with(sym)
            assert isinstance(sym, ast.VarDecl)
            self.file_sema.collect_newly_resolved(sym)
            return True
        return False

    def visit_UAttr(self, node: ast.UAttr) -> bool:
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

        self.type_checker(new_node, forward_push=False)
        return True

    def visit_UCallMethod(self, node: ast.UCallMethod):
        assert node.obj
        if node.obj.type in (None, _ty.Unk):
            node.obj.scope = node.scope  # type: ignore
            if not self.bind_unresolved(node.obj):
                logger.debug(f"Cannot resolve {
                    node.obj} for node {node}")
                return False

        if node.obj.type in (None, _ty.Unk):
            return False

        assert node.obj.type is not None
        class_symbol = Symbol(
            name=node.obj.type.name, kind=Symbol.Kind.Class)
        class_node = self.sym_tbl.global_scope.get(class_symbol)
        assert class_node, f"Cannot find class {node.obj.type.name}"

        func_symbol = FuncSymbol(node.attr)
        methods: Optional[FuncOverloads] = class_node.symbols.get(
            func_symbol)
        if not methods:
            self.report_error(node, f"No method called {
                node.attr} in class {node.obj.get_type()}")
            return False

        args = tuple([node.obj] + list(node.args))  # self + args

        logger.debug(f"UCallMethod obj: {node.obj}")
        class_type = node.obj.type

        # check if is method
        template_spec = None
        if isinstance(class_type, _ty.CompositeType):
            if class_node.template_params:
                assert len(class_node.template_params) == len(
                    class_type.params)
                template_spec = dict(
                    zip(class_node.template_params, class_type.params))

        logger.debug(f"resolving UCallMethod: template_spec: {
            template_spec}")

        if method_candidates := methods.lookup(args, template_spec):
            logger.debug(f"UCallAttr found method: {
                method_candidates}")
            if len(method_candidates) > 1:
                self.report_error(
                    node, f"Method {node.attr} has more than one candidates")
                return False
            elif not method_candidates:
                return False

            method, method_sig = method_candidates[0]

            new_node = ast.CallMethod(
                obj=node.obj, method=method, args=node.args, loc=node.loc)
            new_node.type = method_sig.output_type

            node.replace_all_uses_with(new_node)
            self.type_checker(new_node, forward_push=False)
            self.file_sema.collect_newly_resolved(new_node)
            return True
        else:
            self.report_error(node, f"Method {node.attr} not found")
            return False

    def visit_UClass(self, node: ast.UClass):
        logger.debug(f"TODO Bind unresolved class {node}")
        return False
