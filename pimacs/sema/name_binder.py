from typing import Any, Dict, Optional, Tuple

import pimacs.ast.type as _ty
from pimacs.logger import get_logger

from . import ast
from .func import FuncOverloads
from .utils import FuncSymbol, Symbol

logger = get_logger(__name__)


class NameBinder:
    '''
    NameBinder is a helper class to bind unresolved names in the AST.
    It requires the symbol-table is built before the binding, and the type-checker is ready to resolve the types.
    '''

    def __init__(self, file_sema: Any, type_checker: Any):
        self.type_checker = type_checker
        self.file_sema = file_sema

        self.name_to_module: Dict[str, ast.Module] = {}

    def report_error(self, node: ast.Node, message: str):
        self.file_sema.report_error(node, message)

    def collect_newly_resolved(self, node: ast.Node):
        self.file_sema.collect_newly_resolved(node)

    @property
    def sym_tbl(self):
        return self.file_sema.sym_tbl

    def __call__(self, node: ast.Node) -> bool:
        assert node.is_resolved() is False, f"Node {node} is already resolved"
        return self.bind_unresolved(node)

    def bind_unresolved(self, node: ast.Node) -> bool:
        if node.is_resolved():
            return True

        method_name = f'visit_{type(node).__name__}'
        if method := self.__dict__.get(method_name):
            return getattr(self, method_name)(node)
        return getattr(self, method_name)(node)

    def visit_VarRef(self, node: ast.VarRef) -> bool:
        assert node.target and not node.target.is_resolved()
        if self.bind_unresolved(node.target):
            self.collect_newly_resolved(node)
            return True
        return False

    def visit_VarDecl(self, node: ast.VarDecl) -> bool:
        assert node.init and not node.init.is_resolved()
        if self.bind_unresolved(node.init):
            self.collect_newly_resolved(node)
            return True
        return False

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

        if _ty.Unk in call.type_spec:
            # The call's type spec is not resolved yet
            return False

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
        symbol = Symbol(name=node.name, kind=Symbol.Kind.Var)

        if sym := node.scope.get(symbol):
            node.replace_all_uses_with(sym)
            assert isinstance(sym, ast.VarDecl)
            self.collect_newly_resolved(sym)
            return True
        return False

    def visit_UAttr(self, node: ast.UAttr) -> bool:
        if not node.value.is_resolved():
            if isinstance(node.value, ast.UVarRef):
                # UVarRef.scope is left None, since it is created in Lark parser.
                node.value.scope = node.scope
            if not self.bind_unresolved(node.value):
                return False

        attr_name = node.attr  # type: ignore
        if node.value.get_type() in (None, _ty.Unk):  # type: ignore
            return False

        if isinstance(node.value, ast.Module):
            return self.process_UAttr_module_attr(node, attr_name)

        else:
            class_symbol = Symbol(
                name=node.value.get_type().name, kind=Symbol.Kind.Class)  # type: ignore
            class_node = self.sym_tbl.global_scope.get(class_symbol)
            assert class_node, f"Class {class_symbol.name} not found in {node}"

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

    def process_UAttr_module_attr(self, node: ast.UAttr, attr_name: str) -> bool:
        # get an attribute from a module
        assert False

    def visit_UCallMethod(self, node: ast.UCallMethod):
        assert node.obj
        if node.obj.type in (None, _ty.Unk):
            node.obj.scope = node.scope  # type: ignore
            if not self.bind_unresolved(node.obj):
                return False
            # node.obj is resolved, and we can continue to resolve the UCallMethod.

        assert node.obj.type is not None

        if not isinstance(node.obj, ast.Module):
            module = node.obj.type.module if node.obj.type.module else self.file_sema._cur_module
            assert module

            class_symbol = Symbol(
                name=node.obj.type.name, kind=Symbol.Kind.Class)
            # retrieve the class from the module
            if class_node := module.ctx.symbols.global_scope.get(class_symbol):
                return self.process_class_call_method(node, class_node)

        else:
            module_symbol = Symbol(
                name=node.obj.name, kind=Symbol.Kind.Module)
            return self.process_module_call_method(node, node.obj)
        return False

    def _bind_method(self, node: ast.UCallMethod,
                     methods: FuncOverloads, args,
                     type_spec: Optional[Dict[str, _ty.Type]] | Tuple[_ty.Type, ...]) -> Optional[ast.CallMethod]:
        """ Bind the method to the node. """

        assert node.obj
        assert node.obj.type is not None
        assert methods

        if method_candidates := methods.lookup(args, template_spec=type_spec):

            logger.debug(f"UCallAttr found method: {
                method_candidates}")
            if len(method_candidates) > 1:
                self.report_error(
                    node, f"Method {node.attr} has more than one candidates")
                return None
            elif not method_candidates:
                return None

            method, method_sig = method_candidates[0]

            new_node = ast.CallMethod(
                obj=node.obj, method=method, args=node.args, loc=node.loc)
            new_node.type = method_sig.output_type

            node.replace_all_uses_with(new_node)

            self.type_checker(new_node, forward_push=False)
            self.file_sema.collect_newly_resolved(new_node)
            return new_node
        else:
            self.report_error(node, f"Function {node.attr} not found")
            return None

    def process_module_call_method(self, node: ast.UCallMethod, module_node: ast.Module) -> bool:
        '''
        This method is called when the obj is a module, e.g. `math.sin()`
        '''
        assert module_node.ctx
        methods = module_node.ctx.symbols.global_scope.get_local(
            FuncSymbol(node.attr))

        # TODO: Unify the following code with class method
        if new_node := self._bind_method(node, methods, node.args, node.type_spec):
            new_node.method.module_name = module_node.name
            return True
        return False

    def process_class_call_method(self, node: ast.UCallMethod, class_node: ast.AnalyzedClass) -> bool:
        ''' This method is called when the obj is a class, e.g. `App.run()` `'''
        assert node.obj
        assert node.obj.type is not None

        logger.debug(f"processing class call method {node}")

        func_symbol = FuncSymbol(node.attr)
        methods: Optional[FuncOverloads] = class_node.symbols.get(
            func_symbol)
        if not methods:
            self.report_error(node, f"No method called {
                node.attr} in class {node.obj.get_type()}")
            return False

        args = tuple([node.obj] + list(node.args))  # self + args
        class_type = node.obj.type

        # check if is method
        template_spec = None
        if isinstance(class_type, _ty.CompositeType):
            if class_node.template_params:
                if len(class_node.template_params) != len(
                        class_type.params):
                    self.report_error(node, f"Template params mismatch: {
                                      class_node.template_params} vs {class_type.params}")
                template_spec = dict(
                    zip(class_node.template_params, class_type.params))

        logger.debug(f"resolving UCallMethod with template_spec: {
                     template_spec}")

        ret = self._bind_method(node, methods, args,
                                template_spec)  # type: ignore
        return bool(ret)

    def visit_UClass(self, node: ast.UClass):
        logger.debug(f"TODO Bind unresolved class {node}")
        return False

    def visit_UModule(self, node: ast.UModule) -> ast.Node | None:
        # TODO: This path is deprecated
        if sym := node.scope.get(Symbol(name=node.name, kind=Symbol.Kind.Module)):
            if sym.is_resolved():
                node.replace_all_uses_with(sym)
                self.file_sema.collect_newly_resolved(sym)
                return sym

        # The actual Module is set directly into the name_to_module
        if node.name in self.name_to_module:
            module = self.name_to_module[node.name]
            node.replace_all_uses_with(module)
            self.file_sema.collect_newly_resolved(module)
            return True

        return False
