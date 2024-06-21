from abc import ABC, abstractmethod

import pimacs.ast.ast as ast
from pimacs.sema.ast_visitor import IRMutator


class ASTWalker(ABC):
    @abstractmethod
    def walk_to_node_pre[T](self, node: T) -> bool:  # type: ignore
        ''' This method is called before walking into the children of the node. If it returns False, the children of
        the node will not be visited.'''
        return True

    @abstractmethod
    def walk_to_node_post[T](self, node: T) -> T | None:  # type: ignore
        ''' This method is called after walking into the children of the node. If it returns None, the walk is
        terminated, otherwise, the returned value is sliced in where the old node previously appears. '''
        return node


class Traversal(IRMutator):
    def __init__(self, walker: ASTWalker) -> None:
        self.walker = walker

    def __call__(self, node):
        return self.visit(node)

    def do_it[T](self, node: T | None) -> T | None:  # type: ignore
        if node is None:
            return node

        if not self.walker.walk_to_node_pre(node):
            return node

        node = super().visit(node)

        if node:
            node = self.walker.walk_to_node_post(node)  # type: ignore
        return node

    # This is a trick to check a node itself rather than walk into its children directly.
    def visit_list(self, nodes: list | tuple) -> list | tuple:
        ret = []
        for node in nodes:
            if not self.do_it(node):
                return nodes
            else:
                ret.append(node)
        return ret if isinstance(nodes, list) else tuple(ret)

    def visit_Assign(self, node: ast.Assign):
        if x := self.do_it(node.target):
            node.target = x  # type: ignore
        else:
            return node

        if x := self.do_it(node.value):
            node.value = x  # type: ignore
        else:
            return node

        return node

    def visit_UVarRef(self, node: ast.UVarRef):
        return node

    def visit_Attribute(self, node: ast.Attribute):
        if x := self.do_it(node.value):
            node.value = x
        return node

    def visit_UAttr(self, node: ast.UAttr):
        if x := self.do_it(node.value):
            node.value = x
        return node

    def visit_Select(self, node: ast.Select):
        if x := self.do_it(node.cond):
            node.cond = x
        else:
            return node

        if x := self.do_it(node.then_expr):
            node.then_expr = x
        else:
            return node

        if x := self.do_it(node.else_expr):
            node.else_expr = x
        else:
            return node

        return node

    def visit_CallParam(self, node: ast.CallParam):
        if x := self.do_it(node.value):
            node.value = x
        else:
            return node

        return node

    def visit_VarDecl(self, node: ast.VarDecl):
        if x := self.do_it(node.init):
            node.init = x
        else:
            return node

        decorators = []
        for i, decorator in enumerate(node.decorators):
            if x := self.do_it(decorator):
                decorators.append(x)
            else:
                return node
        node.decorators = tuple(decorators)

        return node

    def visit_Arg(self, node: ast.Arg):
        if x := self.do_it(node.default):
            node.default = x
        return node

    def visit_Literal(self, node: ast.Literal):
        if x := self.do_it(node.value):
            node.value = x
        return node

    def visit_Function(self, node: ast.Function):
        decorators = []
        for i, decorator in enumerate(node.decorators):
            if x := self.do_it(decorator):
                decorators.append(x)
            else:
                return node
        node.decorators = tuple(decorators)

        args = []
        for i, arg in enumerate(node.args):
            if x := self.do_it(arg):
                args.append(x)
            else:
                return node
        node.args = tuple(args)

        if x := self.do_it(node.body):
            node.body = x
        return node

    def visit_Call(self, node: ast.Call):
        if x := self.do_it(node.target):
            node.target = x
        else:
            return node

        args = []
        for i, arg in enumerate(node.args):
            if x := self.do_it(arg):
                args.append(x)
            else:
                return node
        node.args = tuple(args)

        return node

    def visit_Block(self, node: ast.Block):
        stmts = []
        for i, stmt in enumerate(node.stmts):
            if x := self.do_it(stmt):
                stmts.append(x)
            else:
                return node
        node.stmts = tuple(stmts)

        return node

    def visit_BinaryOp(self, node: ast.BinaryOp):
        if x := self.do_it(node.left):
            node.left = x
        else:
            return node

        if x := self.do_it(node.right):
            node.right = x
        else:
            return node

        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if x := self.do_it(node.operand):
            node.operand = x
        return node

    def visit_If(self, node: ast.If):
        if x := self.do_it(node.cond):
            node.cond = x
        else:
            return node

        if x := self.do_it(node.then_branch):
            node.then_branch = x
        else:
            return node

        elif_branches = []
        for cond, block in node.elif_branches:
            if x := self.do_it(cond):
                cond = x
            else:
                return node

            if x := self.do_it(block):
                block = x
            else:
                return node
            elif_branches.append((cond, block))
        node.elif_branches = tuple(elif_branches)

        if x := self.do_it(node.else_branch):
            node.else_branch = x

        return node

    def visit_Return(self, node: ast.Return):
        if x := self.do_it(node.value):
            node.value = x
        return node

    def visit_File(self, node: ast.File):
        stmts = []
        for i, stmt in enumerate(node.stmts):
            if x := self.do_it(stmt):
                stmts.append(x)
            else:
                return node
        node.stmts = stmts

        return node

    def visit_Class(self, node: ast.Class):
        decorators = []
        for i, decorator in enumerate(node.decorators):
            if x := self.do_it(decorator):
                decorators.append(x)
            else:
                return node
        node.decorators = tuple(decorators)

        body = []
        for i, stmt in enumerate(node.body):
            if x := self.do_it(stmt):
                body.append(x)
            else:
                return node
        node.body = tuple(body)

        return node

    def visit_Guard(self, node: ast.Guard):
        if x := self.do_it(node.header):
            node.header = x
        else:
            return node

        if x := self.do_it(node.body):
            node.body = x
        else:
            return node

        return node
