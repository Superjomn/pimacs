from dataclasses import dataclass

from pyimacs.target.elisp_ast import *


class Transform:
    # TODO: use visitor pattern
    # TODO: add cache to avoid unnecessary visit
    def visit(self, node):
        if isinstance(node, LetExpr):
            return self.visit_Let(node)
        elif isinstance(node, Call):
            return self.visit_Call(node)
        elif isinstance(node, Function):
            return self.visit_Function(node)
        elif isinstance(node, Expression):
            return self.visit_Expression(node)
        elif isinstance(node, Var):
            return self.visit_Var(node)
        elif isinstance(node, Symbol):
            return self.visit_Symbol(node)
        elif isinstance(node, Token):
            return self.visit_Token(node)
        elif isinstance(node, IfElse):
            return self.visit_IfElse(node)
        else:
            raise Exception(f"Unknown node type {type(node)}")

    def visit_IfElse(self, node: IfElse):
        cond = self.visit(node.cond)
        then_body = self.visit(node.then_body)
        else_body = self.visit(node.else_body)
        return IfElse(cond, then_body, else_body)

    def visit_Let(self, node: LetExpr):
        new_vars = []
        for var in node.vars:
            new_vars.append(self.visit(var))

        new_body = []
        for expr in node.body:
            new_body.append(self.visit(expr))
        return LetExpr(new_vars, new_body)

    def visit_Call(self, node: Call):
        node.func = self.visit(node.func)
        for no, arg in enumerate(node.args):
            node.args[no] = self.visit(arg)
        return node

    def visit_Function(self, node: Function):
        for id, arg in enumerate(node.args):
            node.args[id] = self.visit(arg)
        node.body = self.visit(node.body)
        return node

    def visit_Expression(self, node: Expression):
        symbols = []
        for id, arg in enumerate(node.symbols):
            new_arg = self.visit(arg)
            symbols.append(new_arg)
        return Expression(*symbols)

    def visit_Var(self, node: Var):
        return node

    def visit_Symbol(self, node: Symbol):
        return node

    def visit_Token(self, node: Token):
        return node


@dataclass
class Simplify(Transform):

    def visit_Let(self, node: LetExpr):
        instant_values = {}
        for arg in node.vars:
            print(arg, arg.user_count)
            for user in arg.users:
                print(' - ', user)
            if arg.user_count <= 1:
                # for return expr, the user_count is 0
                instant_values[arg] = None

        node.vars = filter(lambda x: x not in instant_values, node.vars)

        # get value
        for expr in node.body:
            if isinstance(expr, Expression) and expr.symbols[0] == Symbol('setq') and expr.symbols[1] in instant_values:
                instant_values[expr.symbols[1]] = expr.symbols[2]

        # replace
        for sym, value in instant_values.items():
            assert value, f"symbol {sym} has no value"
            users = [x for x in sym.users]
            for user in users:
                user.replace_symbol(sym, value)

        # clean unnecessary setq
        new_exprs = []
        for expr in node.body:
            if isinstance(expr, Expression) and expr.symbols[0] == Symbol('setq') and expr.symbols[1] in instant_values:
                continue
            if isinstance(expr, Var) and expr in instant_values:
                # for "return" expr, the expression is a single Var
                new_exprs.append(instant_values[expr])
            else:
                new_exprs.append(self.visit(expr))
        node.body = new_exprs
        return node


def simplify(func: Function):
    ''' Simplify code. '''
    Simplify().visit(func)


def transform(func: Function):
    ''' Transforms. '''
    simplify(func)
