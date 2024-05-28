import pimacs.ast.type as _ty
from pimacs.ast import ast
from pimacs.ast.utils import WeakSet
from pimacs.logger import logger
from pimacs.sema.context import ModuleContext

from .ast_visitor import IRVisitor


class TypeInfer(IRVisitor):
    def __init__(self, ctx: ModuleContext):
        self.ctx = ctx
        self._forward_push = False
        # hold the nodes that are newly updated
        self._newly_updated_nodes: WeakSet = WeakSet()

    @property
    def _backward_push(self) -> bool:
        return not self._forward_push

    def report_error(self, node: ast.Node, message: str):
        self.ctx.report_sema_error(node, message)

    def update_type(self, node, type: _ty.Type):
        ''' Lasily update the type of the node and its users. '''
        if is_unk(type):
            return

        node.type = type
        self._newly_updated_nodes.add(node)

    def pollute_users(self, node: ast.Node):
        ''' Pollute the users of the node. '''
        while self._newly_updated_nodes:
            node = self._newly_updated_nodes.pop()
            for user in node.users:
                logger.info(f'Pollute {node}')
                self.visit(user)

    def __call__(self, node: ast.Node, forward_push: bool = True):
        '''
        Args:
            node: The root node of the AST.
            forward_push: Whether to push the type information forward.
        '''
        self._forward_push = forward_push
        self._newly_updated_nodes.clear()

        if self._backward_push:
            self._newly_updated_nodes.add(node)
            self.pollute_users(node)
        else:
            self.visit(node)

    def visit_Call(self, node: ast.Call):
        # visit leaf nodes
        if self._forward_push:
            super().visit_Call(node)

        match type(node.func):
            case ast.UFunction:
                logger.debug(f"visit_Call: UFuntion: {node.func}")
                return
            case ast.Function:
                self.update_type(node, node.func.return_type)  # type: ignore
            case _:
                logger.info(f'Unresolved function call: {node.func}')
                return

    def visit_VarDecl(self, node: ast.VarDecl):
        if self._forward_push:
            super().visit_VarDecl(node)

        if node.init is not None:
            if not (is_unk(node.init.type) or is_unk(node.type)):
                if node.init.type != node.type:
                    self.report_error(node, f'Type mismatch, declared as {
                                      node.type}, but got value of {node.init.type}')
                else:
                    return
            elif not is_unk(node.init.type):
                self.update_type(node, node.init.type)  # type: ignore

    def visit_VarRef(self, node: ast.VarRef):
        if self._forward_push:
            super().visit_VarRef(node)

        if node.target and not is_unk(node.target.type):
            self.update_type(node, node.target.type)  # type: ignore

    def visit_Arg(self, node: ast.Arg):
        if self._forward_push:
            super().visit_Arg(node)

        if not is_unk(node.type):
            if node.default and not is_unk(node.default.type):
                if node.default.type != node.type:
                    self.report_error(node, f'Type mismatch, declared as {
                                      node.type}, but got value of {node.default.type}')
                else:
                    return

        if node.default is not None:
            self.update_type(node, node.default.type)  # type: ignore

    def visit_BinaryOp(self, node: ast.BinaryOp):
        if self._forward_push:
            super().visit_BinaryOp(node)

        if not is_unk(node.type):
            return

        if not (node.left.type_determined and node.right.type_determined):
            return

        if node.left.type != node.right.type:
            self.report_error(node, f'Type mismatch: {
                              node.left.type} != {node.right.type}')
        else:
            self.update_type(node, node.left.type)  # type: ignore

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if self._forward_push:
            super().visit_UnaryOp(node)

        if not is_unk(node.type):
            return

        if not node.operand.type_determined:
            return

        self.update_type(node, node.operand.type)  # type: ignore

    def visit_CallParam(self, node: ast.CallParam):
        if self._forward_push:
            super().visit_CallParam(node)

        if node.value is not None and not is_unk(node.value.type):
            if not is_unk(node.type):
                if node.value.type != node.type:
                    self.report_error(node, f'Type mismatch, declared as {
                                      node.type}, but got value of {node.value.type}')
            else:
                self.update_type(node, node.value.type)  # type: ignore

    def visit_Assign(self, node: ast.Assign):
        if self._forward_push:
            super().visit_Assign(node)

        if not (node.value.type_determined and node.target.type_determined):
            return

        if node.value.type != node.target.type:
            self.report_error(node, f'Type mismatch: {
                              node.value.type} != {node.target.type}')

    def visit_Attribute(self, node: ast.Attribute):
        if self._forward_push:
            super().visit_Attribute(node)

        if not is_unk(node.type):
            return

        if is_unk(node.type):
            self.update_type(node, node.value.type)  # type: ignore

    def visit_str(self, node: str):
        pass


def is_unk(type: _ty.Type | None) -> bool:
    return type in (_ty.Unk, None)
