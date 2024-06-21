from typing import *

from multimethod import multimethod

import pimacs.ast.type as _ty
from pimacs.ast import ast
from pimacs.ast.utils import WeakSet
from pimacs.logger import logger
from pimacs.sema.context import ModuleContext

from .ast_visitor import IRVisitor
from .ast_walker import ASTWalker, Traversal


class TypeChecker(IRVisitor):
    '''
    TypeChecker helps to infer the type of the AST nodes. It performs a bottom-up type inference first, and for the newly
    type-updated nodes, it will pollute the users of the nodes.
    '''

    def __init__(self, ctx: ModuleContext):
        self.ctx = ctx
        self.to_push_forward = False
        # hold the nodes that are newly updated
        self._newly_updated_nodes: WeakSet = WeakSet()

    @multimethod
    def convert_type(self, source: _ty.Type | None, target: _ty.Type) -> _ty.Type | None:
        '''
        Convert source to target. If the conversion is not possible, return None.

        This method could be used to verify if a value could be pass to a argument.
        '''
        if source is None:
            return target
        if target is None:
            return None

        if source == target:
            return source

        if source == _ty.Int and target == _ty.Float:
            return target
        if source == _ty.Float and target == _ty.Int:
            return target

        if source == _ty.Bool and target == _ty.Int:
            return target

        # Currently, the LispType works as an Any type
        # TODO: Add an dedicated Any type
        if source == _ty.LispType:
            return True

        # TODO: Any class types could be converted to bool if the method __bool__ is defined.
        if source.get_nosugar_type() == target.get_nosugar_type():
            # Convert Value to Value?
            if target.is_optional:
                return True
        if target.is_optional and source == _ty.Nil:
            return True

        return None

    @multimethod  # type: ignore
    def convert_type(self, source: List[_ty.Type], target: _ty.Type) -> List[_ty.Type]:
        return [self.convert_type(s, target) for s in source]

    def is_type_numeric(self, type: _ty.Type) -> bool:
        return type in (_ty.Int, _ty.Float)

    @property
    def to_push_backward(self) -> bool:
        return not self.to_push_forward

    def report_error(self, node: ast.Node, message: str):
        self.ctx.report_sema_error(node, message)

    def update_type(self, node, type: _ty.Type):
        ''' Lasily update the type of the node and its users. '''
        if is_unk(type):
            return

        node.type = type
        self._newly_updated_nodes.add(node)

    def _pollute_users(self, node: ast.Node):
        ''' Pollute the users of the node. '''
        while self._newly_updated_nodes:
            node = self._newly_updated_nodes.pop()
            for user in node.users:
                logger.debug(f'Pollute {node}')
                self.visit(user)

    def __call__(self, node: ast.Node, forward_push: bool = True):
        '''
        Args:
            node: The root node of the AST.
            forward_push: Whether to push the type information forward.
        '''
        self.to_push_forward = forward_push
        self._newly_updated_nodes.clear()

        if self.to_push_backward:
            self._newly_updated_nodes.add(node)
            self._pollute_users(node)
        else:
            self.visit(node)

    def visit_File(self, node: ast.File):
        if self.to_push_forward:
            super().visit_File(node)

    def visit_Call(self, node: ast.Call):
        # visit leaf nodes
        if self.to_push_forward:
            super().visit_Call(node)

        if not is_unk(node.type):
            return

        match type(node.target):
            case ast.UFunction:
                logger.debug(f"visit_Call: UFuntion: {node.target}")
                return
            case ast.Function:
                self.update_type(node, node.target.return_type)  # type: ignore
            case _:
                logger.debug(f'Unresolved function call: {node.target}')
                return

    def visit_VarDecl(self, node: ast.VarDecl):
        if self.to_push_forward:
            super().visit_VarDecl(node)

        if node.init is not None:
            if not (is_unk(node.init.type) or is_unk(node.type)):
                if not self.convert_type(node.init.type, node.type):
                    self.report_error(node, f'Type mismatch, declared as {
                                      node.type}, but got value of {node.init.type}')
                else:
                    return
            elif not is_unk(node.init.type):
                self.update_type(node, node.init.type)  # type: ignore

    def visit_VarRef(self, node: ast.VarRef):
        if self.to_push_forward:
            super().visit_VarRef(node)

        if node.target and not is_unk(node.target.type):
            self.update_type(node, node.target.type)  # type: ignore

    def visit_Arg(self, node: ast.Arg):
        if self.to_push_forward:
            super().visit_Arg(node)

        if not is_unk(node.type):
            if node.default and not is_unk(node.default.type):
                if not self.convert_type(node.default.type, node.type):
                    self.report_error(node, f'Type mismatch, declared as {
                                      node.type}, but got value of {node.default.type}')
                else:
                    return

        if node.default is not None:
            self.update_type(node, node.default.type)  # type: ignore

    def visit_BinaryOp(self, node: ast.BinaryOp):
        if self.to_push_forward:
            super().visit_BinaryOp(node)

        if not is_unk(node.type):
            return

        if not (node.left.type_determined and node.right.type_determined):
            return

        op_type = self.convert_type(node.left.type, node.right.type) or self.convert_type(
            node.right.type, node.left.type)
        if not op_type:
            self.report_error(node, f'Type mismatch: {
                              node.left.type} != {node.right.type}')
        else:
            self.update_type(node, op_type)  # type: ignore

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if self.to_push_forward:
            super().visit_UnaryOp(node)

        if not is_unk(node.type):
            return

        if not node.operand.type_determined:
            return

        self.update_type(node, node.operand.type)  # type: ignore

    def visit_CallParam(self, node: ast.CallParam):
        if self.to_push_forward:
            super().visit_CallParam(node)

        if node.value is not None and not is_unk(node.value.type):
            if not is_unk(node.type):
                if node.value.type != node.type:
                    self.report_error(node, f'Type mismatch, declared as {
                                      node.type}, but got value of {node.value.type}')
            else:
                self.update_type(node, node.value.type)  # type: ignore

    def visit_Assign(self, node: ast.Assign):
        if self.to_push_forward:
            super().visit_Assign(node)

        if not (node.value.type_determined and node.target.type_determined):
            return

        if node.value.type != node.target.type:
            self.report_error(node, f'Type mismatch: {
                              node.value.type} != {node.target.type}')

    def visit_Attribute(self, node: ast.Attribute):
        if self.to_push_forward:
            super().visit_Attribute(node)

        if not is_unk(node.type):
            return

        if is_unk(node.type):
            self.update_type(node, node.value.type)  # type: ignore

    def visit_str(self, node: str):
        pass


def is_unk(type: _ty.Type | None) -> bool:
    return type in (_ty.Unk, None)


@multimethod
def amend_placeholder_types(node: ast.Node):
    '''
    Replace the mistakenly marked type Generic["T"] with PlaceholderType["T"] in Class and Function template params.
    This should be called before Sema, and after the AST is parsed.
    '''
    class Walker(ASTWalker):
        def __init__(self):
            self.visited_nodes: Set[ast.Class | ast.Function] = set()

        def walk_to_node_pre(self, node) -> bool:
            if isinstance(node, (ast.Function, ast.Class)) and node.template_params:
                if node in self.visited_nodes:
                    return False

                mapping = {}
                for param in node.template_params:
                    assert isinstance(param, _ty.PlaceholderType)
                    mapping[_ty.GenericType(param.name)] = param
                amend_placeholder_types([node], mapping)

                self.visited_nodes.add(node)  # avoid duplicate visiting

            return True

        def walk_to_node_post(self, node):
            return node

    Traversal(Walker()).visit(node)


@multimethod  # type: ignore
def amend_placeholder_types(node: ast.Node | List[ast.Node], template_params: Dict[_ty.Type, _ty.Type]):
    '''
    Replace the mistakenly marked type Generic["T"] with PlaceholderType["T"].
    This should be called before Sema.

    Args:
        node: The root node of the AST, it should be a templated class or function.
        template_params: The mapping from the template parameter name to the actual type.
    '''
    class Walker(ASTWalker):
        def overwrite_type(self, type) -> _ty.Type | List[_ty.Type]:
            if isinstance(type, list):
                return [self.overwrite_type(t) for t in type]
            # deal with optional, such as T?
            if type and type.get_nosugar_type() in template_params:
                # template_params should be nosugar, so it is safe to map
                type_param = template_params[type.get_nosugar_type()]
                return type_param.get_optional_type() if type.is_optional else type_param
            if isinstance(type, _ty.CompositeType):
                return type.replace_with(template_params)
            return type

        def walk_to_node_pre(self, node) -> bool:  # type: ignore
            if hasattr(node, "type"):
                node.type = self.overwrite_type(node.type)
            if hasattr(node, "return_type"):
                node.return_type = self.overwrite_type(node.return_type)
            return True

        def walk_to_node_post(self, node):  # type: ignore
            return node

    Traversal(Walker()).visit(node)
