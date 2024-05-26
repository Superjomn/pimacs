import pimacs.ast.type as _ty
from pimacs.ast import ast
from pimacs.logger import logger
from pimacs.sema.ast_visitor import IRVisitor
from pimacs.sema.context import ModuleContext


class TypeInference(IRVisitor):
    def __init__(self, ctx: ModuleContext):
        self.ctx = ctx
        self._force_update: bool = False

    def report_error(self, node: ast.Node, message: str):
        self.ctx.report_sema_error(node, message)

    def update_type(self, node, type: _ty.Type):
        ''' Lasily update the type of the node and its users. '''
        if type is _ty.Unk:
            return

        if not self._force_update:
            if node.type == type:
                return
        else:
            self._force_update = False  # switch off for the next node

        node.type = type
        self._visit_users(node)

    def __call__(self, node: ast.Node, force_update=False):
        '''
        Args:
            node: The root node of the AST.
            force_update: If True, force update the type of the `node`.
        '''
        self._force_update = force_update

        self.visit(node)

    def visit_Call(self, node: ast.Call):
        if node.sema_failed:
            return
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
        if node.sema_failed:
            return
        super().visit_VarDecl(node)

        if node.init is not None:
            if node.init.type is not _ty.Unk:
                if node.type is not _ty.Unk:
                    if node.init.type != node.type:
                        self.report_error(node, f'Type mismatch, declared as {
                                          node.type}, but got value of {node.init.type}')
                else:
                    self.update_type(node, node.init.type)  # type: ignore

    def visit_VarRef(self, node: ast.VarRef):
        if node.sema_failed:
            return
        super().visit_VarRef(node)

        if node.target is not None:
            self.update_type(node, node.target.type)

    def visit_Arg(self, node: ast.Arg):
        if node.sema_failed:
            return
        super().visit_Arg(node)

        if node.default is not None:
            self.update_type(node, node.default.type)  # type: ignore

    def visit_BinaryOp(self, node: ast.BinaryOp):
        if node.sema_failed:
            return
        super().visit_BinaryOp(node)

        if not (node.left.type_determined and node.right.type_determined):
            return

        if node.left.type != node.right.type:
            self.report_error(node, f'Type mismatch: {
                              node.left.type} != {node.right.type}')
        else:
            self.update_type(node, node.left.type)  # type: ignore

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if node.sema_failed:
            return
        super().visit_UnaryOp(node)

        if not node.operand.type_determined:
            return

        self.update_type(node, node.operand.type)  # type: ignore

    def visit_CallParam(self, node: ast.CallParam):
        if node.sema_failed:
            return
        super().visit_CallParam(node)

        if node.value is not None:
            self.update_type(node, node.value.type)  # type: ignore

    def visit_Assign(self, node: ast.Assign):
        if node.sema_failed:
            return
        super().visit_Assign(node)

        if not (node.value.type_determined and node.target.type_determined):
            return

        if node.value.type != node.target.type:
            self.report_error(node, f'Type mismatch: {
                              node.value.type} != {node.target.type}')

    def visit_str(self, node: str):
        pass

    def _visit_users(self, node: ast.Node):
        for user in node.users:
            self.visit(user)
