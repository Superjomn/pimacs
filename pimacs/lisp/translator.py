"""
This file defines a translator that translates the AST to Lisp AST.
"""

from contextlib import contextmanager
from typing import List, Optional, Tuple

from multimethod import multimethod

import pimacs.lisp.ast as lisp_ast
import pimacs.sema.ast as ast
import pimacs.sema.ast_visitor as ast_visitor
from pimacs.sema.context import ModuleContext


class LispTranslator(ast_visitor.IRMutator):
    def __init__(self, ctx: ModuleContext):
        self.ctx = ctx
        self._cur_class: Optional[ast.AnalyzedClass] = None

    def __call__(self,  node) -> lisp_ast.Node:
        return self.visit(node)

    @contextmanager
    def class_scope_guard(self, class_node: ast.AnalyzedClass):
        self._cur_class = class_node
        yield
        self._cur_class = None

    def visit_File(self, node: ast.File) -> lisp_ast.Module:
        ret = lisp_ast.Module(
            name="<unk-file-name>",
            stmts=[],
        )
        for stmt in node.stmts:
            if isinstance(stmt, ast.VarDecl):
                ret.stmts.append(
                    lisp_ast.Assign(lisp_ast.VarRef(stmt.name),
                                    self.visit(stmt.init))
                )
            else:
                ret.stmts.append(self.visit(stmt))

        ret.loc = node.loc
        return ret

    def visit_VarDecl(self, node: ast.VarDecl) -> lisp_ast.Assign | None:
        if node.init:
            ret = lisp_ast.Assign(
                target=lisp_ast.VarRef(node.name),
                value=self.visit(node.init)
            )
            ret.loc = node.loc
            return ret

        return None

    def visit_block(self, node: List[ast.Node] | Tuple[ast.Node]):
        ''' Visit stmts as a block '''
        # collect the VarDecls in the block
        var_decls = set()
        for stmt in node:
            if isinstance(stmt, ast.VarDecl):
                var_decls.add(stmt.name)

        let = lisp_ast.Let(vars=[lisp_ast.VarDecl(name=var) for var in var_decls],
                           body=[self.visit(stmt) for stmt in node])
        let.loc = node[0].loc
        return let

    def visit_Block(self, node: ast.Block) -> lisp_ast.Let:
        ret = self.visit_block(list(node.stmts))
        ret.loc = node.loc
        return ret

    def visit_Function(self, node: ast.Function) -> lisp_ast.Function:
        name = self.get_mangled_name(
            node) if not self._cur_class else self.get_mangled_name(node, self._cur_class)

        ret = lisp_ast.Function(
            name=name,
            args=[self.visit(arg) for arg in node.args],
            body=self.visit(node.body),
        )
        ret.loc = node.loc
        return ret

    def visit_Literal(self, node: ast.Literal) -> lisp_ast.Literal:
        ret = lisp_ast.Literal(
            value=node.value
        )
        ret.loc = node.loc
        return ret

    op_mapping = {
        ast.BinaryOperator.ADD: '+',
        ast.BinaryOperator.SUB: '-',
        ast.BinaryOperator.MUL: '*',
        ast.BinaryOperator.DIV: '/',
        ast.BinaryOperator.MOD: '%',
        ast.BinaryOperator.EQ: '=',
        ast.BinaryOperator.NE: '/=',
        ast.BinaryOperator.LT: '<',
        ast.BinaryOperator.LE: '<=',
        ast.BinaryOperator.GT: '>',
        ast.BinaryOperator.GE: '>=',
        ast.BinaryOperator.AND: 'and',
        ast.BinaryOperator.OR: 'or',
        ast.UnaryOperator.NOT: 'not',
    }

    def visit_BinaryOp(self, node: ast.BinaryOp) -> lisp_ast.List:
        ret = lisp_ast.List(
            elements=[
                LispTranslator.op_mapping[node.op],
                self.visit(node.left),
                self.visit(node.right),
            ]
        )
        ret.loc = node.loc
        return ret

    def visit_UnaryOp(self, node: ast.UnaryOp):
        ret = lisp_ast.List(
            elements=[
                LispTranslator.op_mapping[node.op],
                self.visit(node.operand),
            ]
        )
        ret.loc = node.loc
        return ret

    def visit_VarRef(self, node: ast.VarRef) -> lisp_ast.VarRef:
        name = node.name if node.name else node.target.name  # type: ignore
        ret = lisp_ast.VarRef(name=name)
        ret.loc = node.loc
        return ret

    def visit_Arg(self, node: ast.Arg):
        ret = lisp_ast.VarRef(
            name=node.name,
        )
        ret.loc = node.loc
        return ret

    def visit_If(self, node: ast.If) -> lisp_ast.If:
        if not node.elif_branches:
            ret = lisp_ast.If(
                cond=self.visit(node.cond),
                then_block=self.visit(node.then_branch),
                else_block=self.visit(node.else_branch)
            )
            ret.loc = node.loc
            return ret
        else:
            branches = []
            branches.append((node.cond, node.then_branch))
            for cond, body in node.elif_branches:
                branches.append((cond, body))
            if node.else_branch:
                branches.append((None, node.else_branch))  # type: ignore

            root_if: lisp_ast.If | None = None
            last_if: lisp_ast.If | None = None
            for (cond, body) in branches:
                if cond is None:
                    last_if.else_block = self.visit(body)
                else:
                    new_if = lisp_ast.If(
                        cond=self.visit(cond),
                        then_block=self.visit(body),
                        else_block=None)
                    new_if.loc = cond.loc

                    if root_if is None:
                        root_if = new_if
                        last_if = new_if
                    else:
                        assert last_if
                        last_if.else_block = new_if
            assert root_if
            return root_if

    def visit_While(self, node: ast.While) -> lisp_ast.While:
        ret = lisp_ast.While(
            cond=self.visit(node.cond),
            body=self.visit(node.body)
        )
        ret.loc = node.loc
        return ret

    def visit_CallParam(self, node: ast.CallParam):
        if node.name:
            ret = lisp_ast.VarRef(name=node.name)
            ret.loc = node.loc
        else:
            assert node.value
            ret = self.visit(node.value)
            ret.loc = node.loc
        return ret

    def visit_Call(self, node: ast.Call) -> lisp_ast.List:
        assert isinstance(node.target, ast.Function), f"Expected function, got {
            node.target}"
        func_name = self.get_mangled_name(node.target)
        ret = lisp_ast.List(
            elements=[
                func_name] + [self.visit(arg) for arg in node.args]
        )
        ret.loc = node.loc
        return ret

    def visit_CallMethod(self, node: ast.CallMethod) -> lisp_ast.List:
        obj = node.obj
        class_name = obj.type.name  # type: ignore
        class_node = self.ctx.get_symbol(
            ast.Symbol(ast.Symbol.Kind.Class, class_name))  # type: ignore
        assert class_node, f"Class {class_name} not found"
        func_name = self.get_mangled_name(
            node.target, class_node)  # type: ignore
        ret = lisp_ast.List(
            elements=[
                func_name] + self.visit_list(node.args))  # type: ignore
        ret.loc = node.loc
        return ret

    def visit_LispCall(self, node: ast.LispCall) -> lisp_ast.List:
        ret = lisp_ast.List(
            elements=[node.target] + [self.visit(arg) for arg in node.args]
        )
        ret.loc = node.loc
        return ret

    def visit_Return(self, node: ast.Return) -> lisp_ast.Return:
        ret = lisp_ast.Return(
            value=self.visit(node.value)
        )
        ret.loc = node.loc
        return ret

    def visit_Assign(self, node: ast.Assign) -> lisp_ast.Assign:
        ret = lisp_ast.Assign(
            target=self.visit(node.target),
            value=self.visit(node.value)
        )
        ret.loc = node.loc
        return ret

    def visit_Attribute(self, node: ast.Attribute) -> lisp_ast.Attribute:
        assert node.value.type
        ret = lisp_ast.Attribute(
            class_name=node.value.type.name,
            target=self.visit(node.value),  # type: ignore
            attr=node.attr
        )
        ret.loc = node.loc
        return ret

    def visit_AnalyzedClass(self, node: ast.AnalyzedClass):
        members = node.symbols.get_local(ast.Symbol.Kind.Member)
        var_decls = []
        for member in members:
            var_decls.append(lisp_ast.VarDecl(name=member.name,
                                              init=self.visit(member.init)
                                              ))
        ret = lisp_ast.Struct(
            name=node.name,
            fields=var_decls
        )

        with self.class_scope_guard(node):
            methods = self.visit_class_methods(node)
            ret.methods = methods

        ret.loc = node.loc

        return ret

    def visit_class_methods(self, node: ast.AnalyzedClass):
        methods = node.symbols.get_local(ast.Symbol.Kind.Func)
        with self.class_scope_guard(node):
            func_decls = []
            for method in methods:
                if method.name != '__init__':
                    # The __init__ is converted to constructor functions in the global scope during Sema
                    func_decls.append(self.visit(method))
            return func_decls

    def visit_Select(self, node: ast.Select):
        ret = lisp_ast.List(elements=[
            lisp_ast.VarRef("if"),
            self.visit(node.cond),
            self.visit(node.then_expr),
            self.visit(node.else_expr)
        ])
        ret.loc = node.loc
        return ret

    def visit_Guard(self, node: ast.Guard):
        ret = lisp_ast.Guard(
            header=self.visit(node.header),
            body=self.visit(node.body)
        )
        ret.loc = node.loc
        return ret

    def visit_MakeObject(self, node: ast.MakeObject):
        assert node.type
        class_name = node.type.name
        list_elements = [lisp_ast.VarRef(f"make-{class_name}")]

        class_node = self.get_class(class_name)
        for member in class_node.symbols.get_local(ast.Symbol.Kind.Member):
            member_name = member.name
            member_value = self.visit(
                member.init) if member.init else lisp_ast.VarRef("nil")
            list_elements.append(lisp_ast.VarRef(f":{member_name}"))
            list_elements.append(member_value)

        ret = lisp_ast.List(elements=list_elements)  # type: ignore
        ret.loc = node.loc
        return ret

    def get_class(self, name: str) -> ast.AnalyzedClass:
        class_node = self.ctx.get_symbol(
            ast.Symbol(name, ast.Symbol.Kind.Class))
        assert class_node, f"Class {name} not found"
        return class_node

    @multimethod
    def get_mangled_name(self, class_node: ast.AnalyzedClass) -> str:
        # TODO: Consider module scope later
        return class_node.name

    @multimethod  # type: ignore
    def get_mangled_name(self, func: ast.Function) -> str:
        arg_type_list = '_'.join([str(arg.type) for arg in func.args])
        return f"{func.name}--{arg_type_list}"

    @multimethod  # type: ignore
    def get_mangled_name(self, func: ast.Function, class_node: ast.AnalyzedClass) -> str:
        arg_type_list = '_'.join([str(arg.type) for arg in func.args])
        return f"{class_node.name}--{func.name}--{arg_type_list}"
