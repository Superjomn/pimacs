import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import *

import lark
from lark import Lark, Transformer
from lark.indenter import PythonIndenter

import pimacs.lang.ir as ir
import pimacs.lang.type as _type
from pimacs.lang.context import ModuleCtx, Scope, Symbol, SymbolTable
from pimacs.lang.ir_visitor import IRMutator, IRVisitor
from pimacs.lang.sema import Sema, catch_sema_error, report_sema_error


def get_lark_parser():
    dsl_grammar = open(os.path.join(
        os.path.dirname(__file__), 'grammar.g')).read()
    return Lark(dsl_grammar, parser='lalr', postlex=PythonIndenter())


def get_parser(code:Optional[str]=None, filename: str = "<pimacs>"):
    source : ir.PlainCode | ir.FileName = ir.PlainCode(code) if code else ir.FileName(filename)
    dsl_grammar = open(os.path.join(
        os.path.dirname(__file__), 'grammar.g')).read()
    return Lark(dsl_grammar, parser='lalr', postlex=PythonIndenter(),
                transformer=PimacsTransformer(source=source))


def parse(code: str | None = None, filename: str = "<pimacs>", build_ir=True):
    if code:
        source = ir.PlainCode(code)
        parser = get_parser(code=code)
    else:
        code = open(filename).read()
        source = ir.FileName(filename) # type: ignore
        parser= get_parser(code=None, filename=filename)

    stmts = parser.parse(code)
    the_ir = ir.File(stmts=stmts, loc=ir.Location(source, 0, 0))
    if build_ir:
        the_ir = BuildIR().visit(the_ir)
    return the_ir


class PimacsTransformer(Transformer):
    def __init__(self, source: ir.PlainCode | ir.FileName):
        self.source = source

    def file_input(self, items):
        return items

    def func_def(self, items):
        name: str = items[0].value
        args: List[ir.ArgDecl] = safe_get(items, 1, [])
        type = safe_get(items, 2, None)
        block: ir.Block = safe_get(items, 3, None)
        loc = ir.Location(self.source, items[0].line, items[0].column)

        if block is None:
            raise Exception(f"{loc}:\nFunction {name} must have a block")

        return ir.FuncDecl(name=name, args=args, body=block,
                           return_type=type,
                           loc=loc)

    def func_args(self, items):
        return items

    def func_arg(self, items) -> ir.ArgDecl:
        name_token = safe_get(items, 0, None)
        name = name_token.value if name_token else None
        type_ = safe_get(items, 1, None)
        default = safe_get(items, 2, None)

        return ir.ArgDecl(name=name, type=type_, default=default, loc=ir.Location(self.source, items[0].line, items[0].column))


    def type_base(self, items):
        return items[0]

    def type(self, items):
        assert len(items) == 1
        return items[0]

    def set_type(self, items):
        "The type of Set"
        assert len(items) == 1
        return _type.SetType(inner_types=items)

    def dict_type(self, items):
        '''
        Type of Dict.
        '''
        assert len(items) == 2
        return _type.DictType(key_type=items[0], value_type=items[1])

    def list_type(self, items):
        '''
        Type of List.
        '''
        assert len(items) == 1
        return _type.ListType(inner_types=items)


    def PRIMITIVE_TYPE(self, items) -> _type.Type:
        # This terminal is failed, because it is covered by the custom_type
        raise NotImplementedError()

    def custom_type(self, items) -> _type.Type:
        assert len(items) == 1
        ret = _type.parse_primitive_type(items[0].value)
        if ret is not None:
            return ret
        return _type.Type(type_id=_type.TypeId.CUSTOMED, name=items[0].value)

    def block(self, items) -> ir.Block:
        stmts = list(filter(lambda x: x is not None, items))
        doc_string = None
        if isinstance(items[0], ir.DocString):
            doc_string = stmts[0]
            stmts = stmts[1:]

        return ir.Block(stmts=stmts, doc_string=doc_string, loc=stmts[0].loc)


    def return_stmt(self, items) -> ir.ReturnStmt:
        assert len(items) == 2
        loc = ir.Location(self.source, items[0].line, items[0].column)
        if items[1]:
            return ir.ReturnStmt(value=items[1], loc=loc)
        return ir.ReturnStmt(value=None, loc=loc)

    def func_call(self, items) -> ir.FuncCall | ir.LispFuncCall:
        if isinstance(items[0], ir.VarRef):
            name = items[0].name
            loc = items[0].loc
        else:
            name = items[0].value
            loc = ir.Location(self.source, items[0].line, items[0].column)

        if name.startswith('%'):
            assert loc
            func = ir.LispVarRef(name=name, loc=loc)
            args = items[1] if items[1] else []
            return ir.LispFuncCall(func=func, args=args, loc=loc)
        else:
            args = items[1] if items[1] else []
            return ir.FuncCall(func=name, args=args, loc=loc)

    def lisp_name(self, items):
        raise NotImplementedError()

    def lisp_symbol(self, items):
        return ir.LispVarRef(name=items[0].value, loc=ir.Location(self.source, items[0].line, items[0].column))

    def func_call_name(self, items) -> str:
        return items[0]

    def call_param(self, items: List[lark.Tree]) -> ir.CallParam:
        assert False
        assert len(items) == 1
        content = items[0][0]
        return ir.CallParam(name="", value=content, loc=content.loc)

    def call_params(self, items: List[lark.Tree]) -> List[ir.CallParam]:
        for item in items:
            assert isinstance(item, ir.CallParam)
        return items # type: ignore

    def string(self, items: List[lark.lexer.Token]):
        loc = ir.Location(self.source, items[0].line, items[0].column) # type: ignore
        return ir.Constant(value=items[0].value, loc=loc)

    def variable(self, items) -> ir.VarRef:
        assert len(items) == 1
        if isinstance(items[0], ir.VarRef):
            return items[0]
        name = items[0].value
        if name.startswith('%'):
            return ir.LispVarRef(name=name, loc=ir.Location(self.source, items[0].line, items[0].column))
        return ir.VarRef(name=items[0].value, loc=ir.Location(self.source, items[0].line, items[0].column))

    def var_decl(self, items) -> ir.VarDecl:
        loc = ir.Location(self.source, items[0].line, items[0].column)
        name = items[1].value
        type = safe_get(items, 2, None)
        init = safe_get(items, 3, None)
        node =  ir.VarDecl(name=name, type=type, init=init, loc=loc)
        #self._deduce_var_type(node)
        return node

    def let_decl(self, items) -> ir.VarDecl:
        loc = ir.Location(self.source, items[0].line, items[0].column)
        name = items[1].value
        type = safe_get(items, 2, None)
        init = safe_get(items, 3, None)

        node = ir.VarDecl(name=name, type=type, init=init, loc=loc, mutable=False)
        #node = self._deduce_var_type(node)
        return node

    def _deduce_var_type(self, var:ir.VarDecl) -> ir.VarDecl:
        if var.type is not None:
            return var

        if var.init:
            if isinstance(var.init, ir.LispVarRef):
                var.type = _type.LispType
            elif isinstance(var.init, ir.Constant):
                if isinstance(var.init.value, int):
                    var.type = _type.Int
                elif isinstance(var.init.value, float):
                    var.type = _type.Float
                elif isinstance(var.init.value, str):
                    var.type = _type.Str
                elif var.init.value is None:
                    var.type = _type.Nil
                else:
                    raise ValueError(f"{var.loc}\nUnknown constant type {var.init.value}")
            else:
                var.type =  var.init.type # type: ignore

        return var

    def value_param(self, items):
        assert len(items) == 1
        return ir.CallParam(name="", value=items[0], loc=items[0].loc)

    def key_value_param(self, items):
        assert isinstance(items[0], lark.Tree)
        tree: lark.Tree = items[0]
        assert isinstance(tree.children[0], lark.lexer.Token)
        name:str = tree.children[0].value
        value = items[1]
        loc = ir.Location(self.source, tree.children[0].line, tree.children[0].column)
        return ir.CallParam(name=name, value=value, loc=loc)

    def expr(self, items):
        return items[0]

    def number(self, x: List[ir.Constant]):
        assert len(x) == 1
        return x[0]

    def NUMBER(self, x):
        value = float(x) if '.' in x else int(x)
        return ir.Constant(value=value, loc=ir.Location(self.source, x.line, x.column))

    def atom(self, items):
        return items[0]

    def true(self, items):
        return ir.Constant(value=True, loc=None)

    def false(self, items):
        return ir.Constant(value=False, loc=None)

    def nil(self, items):
        return ir.Constant(value=None, loc=None)

    def if_stmt(self, items):
        loc = ir.Location(self.source, items[0].line, items[0].column)
        cond = items[1]
        then_block = items[2]

        elif_blocks = []
        else_block = None
        for i in range(3, len(items)):
            rule: lark.tree.Tree = items[i]
            rule_kind:str = items[i].data

            if rule_kind == 'else_block':
                else_block = rule.children[1]
            elif rule_kind == 'elif_block':
                elif_cond = rule.children[1]
                elif_block = rule.children[2]
                elif_blocks.append((elif_cond, elif_block))

        return ir.IfStmt(cond=cond, then_branch=then_block, elif_branches=elif_blocks, else_branch=else_block, loc=loc)

    def add(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.ADD, loc=items[0].loc)

    def sub(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.SUB, loc=items[0].loc)

    def mul(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.MUL, loc=items[0].loc)

    def div(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.DIV, loc=items[0].loc)

    def le(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.LE, loc=items[0].loc)

    def lt(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.LT, loc=items[0].loc)

    def ge(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.GE, loc=items[0].loc)

    def gt(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.GT, loc=items[0].loc)

    def eq(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.EQ, loc=items[0].loc)

    def ne(self, items):
        return ir.BinaryOp(left=items[0], right=items[1], op=ir.BinaryOperator.NE, loc=items[0].loc)

    def decorator(self, items):
        action = None
        if isinstance(items[0], ir.FuncCall):
            action: ir.FuncCall = items[0]
            loc = action.loc
        else:
            assert isinstance(items[0].value, str)
            action: str = items[0].value
            loc = ir.Location(self.source, items[0].line, items[0].column)

        return ir.Decorator(action=action, loc=loc)

    def decorated(self, items):
        decorators: List[ir.Decorator] = items[:-1]
        func_def: ir.FuncDecl = items[-1]
        func_def.decorators = decorators
        return func_def

    def dotted_name(self, items) -> ir.VarRef:
        return ir.VarRef(name=".".join([x.value for x in items]), loc=ir.Location(self.source, items[0].line, items[0].column))

    def assign_stmt(self, items):
        target = items[0]
        value = items[1]
        return ir.AssignStmt(target=target, value=value, loc=target.loc)

    def class_def(self, items):
        name = items[0].value
        body = items[1]
        loc = ir.Location(self.source, items[0].line, items[0].column)
        return ir.ClassDef(name=name, body=body, loc=loc)

    def class_body(self, items):
        return filter(lambda x: x is not None, items)

    def STRING(self, items):
        assert isinstance(items, str)
        return items

    def doc_string(self, items):
        token = items[0]
        assert token.value.startswith('"') and token.value.endswith('"')
        content = token.value[1:-1]
        return ir.DocString(content=content.strip(), loc=ir.Location(self.source, token.line, token.column))

    def select_expr(self, items):
        true_expr = items[0]
        cond = items[1]
        false_expr = items[2]

        assert isinstance(cond, ir.Expr)
        assert isinstance(true_expr, ir.Expr)
        assert isinstance(false_expr, ir.Expr)
        return ir.SelectExpr(cond=cond, then_expr=true_expr, else_expr=false_expr, loc=cond.loc)

    def guard_stmt(self, items):
        func_call = items[0]
        body = items[1]

        return ir.GuardStmt(header=func_call, body=body, loc=func_call.loc)

    def not_cond(self, items):
        loc = ir.Location(self.source, items[0].line, items[0].column)
        assert isinstance(items[1], ir.Expr)
        return ir.UnaryOp(op=ir.UnaryOperator.NOT, value=items[1], loc=loc)


def safe_get(items, index, default):
    if len(items) > index:
        return items[index]
    return default


class BuildIR(IRMutator):
    '''
    Clean up the symbols in the IR, like unify the VarRef or FuncDecl with the same name in the same scope.
    '''

    # NOTE, the ir nodes should be created when visited.

    def __init__(self):
        self.sym_tbl: SymbolTable = SymbolTable()

    def visit_VarRef(self, node: ir.VarRef):
        assert self.sym_tbl.current_scope.kind != Scope.Kind.Class

        if node.name.startswith('self.'):
            # deal with members
            member = self.sym_tbl.get_symbol(name=node.name[5:], kind=Symbol.Kind.Member)
            if member is None:
                raise KeyError(f"{node.loc}\nMember {node.name} is not declared")
            var = ir.VarRef(decl=member, loc=node.loc) # type: ignore

            obj = self.sym_tbl.get_symbol(name='self', kind=Symbol.Kind.Arg)
            if not obj:
                obj = ir.UnresolvedVarDecl(name='self', loc=node.loc, target_type=_type.Unk) # type: ignore
                catch_sema_error(True, node, f"`self` is not declared")
            return ir.MemberRef(obj=ir.VarRef(decl=obj, loc=node.loc), member=var, loc=node.loc) # type: ignore

        if sym := self.sym_tbl.get_symbol(name = node.name, kind=[Symbol.Kind.Var, Symbol.Kind.Arg]):
            if isinstance(sym, ir.VarDecl):
                var = ir.VarRef(decl=sym, loc=node.loc) # type: ignore
                #self.sym_tbl.add_symbol(Symbol(name=node.name, kind=Symbol.Kind.Var), var)
            elif isinstance(sym, ir.ArgDecl):
                var = ir.VarRef(decl=sym, loc=node.loc)
                #self.sym_tbl.add_symbol(Symbol(name=node.name, kind=Symbol.Kind.Arg), var)
            elif isinstance(sym, ir.VarRef):
                var = sym
            else:
                raise ValueError(f"{node.loc}\nUnknown symbol type {sym}")
            return var
        else:
            raise KeyError(f"{node.loc}\nSymbol {node.name} not found")

    def visit_VarDecl(self, node: ir.VarDecl):
        if self.sym_tbl.current_scope.kind is Scope.Kind.Class:
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Member)
            self.sym_tbl.add_symbol(symbol, node)
        else:
            self.sym_tbl.add_symbol(Symbol(name=node.name, kind=Symbol.Kind.Var), node)

        return node

    def visit_SelectExpr(self, node: ir.SelectExpr):
        node = super().visit_SelectExpr(node)
        #node.cond.add_user(node)
        #node.then_expr.add_user(node)
        #node.else_expr.add_user(node)
        return node

    def visit_FuncDecl(self, node: ir.FuncDecl):
        within_class = self.sym_tbl.current_scope.kind is Scope.Kind.Class
        with self.sym_tbl.scope_guard(kind=Scope.Kind.Func):
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Func)
            assert not self.sym_tbl.contains_locally(symbol), f"{node.loc}\nFunction {node.name} already exists"
            args = node.args if node.args else []
            if within_class:
                # This function is a member function
                if node.is_classmethod:
                    cls_arg = args[0] # cls placeholder
                    assert cls_arg.name == 'cls', f"{node.loc}\nClass method should have the first arg named 'cls'"
                    cls_arg.kind = ir.ArgDecl.Kind.cls_placeholder
                elif not node.is_staticmethod:
                    self_arg = args[0] # self placeholder
                    assert self_arg.name == 'self', f"{node.loc}\nMethod should have the first arg named 'self'"
                    self_arg.kind = ir.ArgDecl.Kind.self_placeholder

            args = [self.visit(arg) for arg in args]

            body = self.visit(node.body)
            return_type = self.visit(node.return_type)
            decorators = [self.visit(decorator) for decorator in node.decorators]
            new_node = ir.FuncDecl(name=node.name, args=args, body=body, return_type=return_type, loc=node.loc, decorators=decorators)
            #for arg in args:
                #arg.add_user(arg)

            # TODO: add users for the decorators

            return self.sym_tbl.add_symbol(symbol, new_node)

    def build_elisp_list(self, items: List[ir.Expr]) -> ir.LispList:
        ''' Lisp list could be a function call, all the raw lisp code should be wrapped in a lisp list.
        A LispList could come from the following cases:
        1. A lisp function call like `%(+ 1 2)`
        2. A pimacs function call with the function name starts with `%`, such as `%message("hello")`
        3. A pimacs function call with the function name come from a lisp variable, such as `message = %message; message("hello")`
        '''
        func = items[0]
        loc = items[0].loc
        assert loc
        if isinstance(func, ir.LispVarRef):
            return ir.LispList(items, loc=loc)

        assert NotImplementedError(f"{loc}\nUnknown lisp function call {func}")
        return ir.LispList(items, loc=loc)

    def visit_ClassDef(self, node: ir.ClassDef):
        assert self.sym_tbl.current_scope.kind == Scope.Kind.Global, f"{node.loc}\nClass should be in the global scope"
        with self.sym_tbl.scope_guard(kind=Scope.Kind.Class):
            symbol = Symbol(name=node.name, kind=Symbol.Kind.Class)
            assert not self.sym_tbl.contains_locally(symbol), f"{node.loc}\nClass {node.name} already exists"
            body = [self.visit(stmt) for stmt in node.body]

            node = ir.ClassDef(name=node.name, body=body, loc=node.loc)
            return self.sym_tbl.add_symbol(symbol, node)

    def visit_FuncCall(self, node: ir.FuncCall):
        with self.sym_tbl.scope_guard():
            func_name = node.func
            assert isinstance(func_name, str)
            func = self.sym_tbl.get_symbol(name=func_name, kind=[Symbol.Kind.Func, Symbol.Kind.Arg, Symbol.Kind.Var])

            if catch_sema_error(not func, node, f"Target function {func_name} not found"):
                return node

            elif isinstance(func, ir.VarDecl):
                true_func = BuildIR.get_true_var(func)
                if isinstance(true_func, ir.LispVarRef):
                    if isinstance(true_func, ir.LispVarRef):
                        return ir.LispFuncCall(func=true_func, args=node.args, loc=node.loc)

                raise ValueError(f"{node.loc}\nUnknown lisp function call {func}")

            else:
                assert func is not None
                new_node =  ir.FuncCall(func=func, args=node.args, loc=node.loc) # type: ignore
                return new_node

    @staticmethod
    def get_true_var(node: ir.VarDecl | ir.VarRef) -> ir.VarDecl | ir.LispVarRef | ir.ArgDecl | ir.UnresolvedVarDecl | ir.Expr:
        if isinstance(node, ir.VarDecl) and node.init:
            return node.init
        if isinstance(node, ir.VarRef):
            assert node.decl
            return node.decl
        if isinstance(node, ir.LispVarRef):
            return node

        return node

    def visit_LispFuncCall(self, node: ir.LispFuncCall):
        return super().visit_LispFuncCall(node)

    def visit_Block(self, node: ir.Block):
        with self.sym_tbl.scope_guard():
            return super().visit_Block(node)

    def visit_ArgDecl(self, node: ir.ArgDecl):
        assert self.sym_tbl.current_scope.kind == Scope.Kind.Func, f"{node.loc}\nArgDecl should be in a function"
        # check type
        if node.kind not in (ir.ArgDecl.Kind.cls_placeholder, ir.ArgDecl.Kind.self_placeholder):
            if not node.type:
                if not node.default:
                    raise ValueError(f"{node.loc}\nArg {node.name} should have a type or a default value")

        symbol = Symbol(name=node.name, kind=Symbol.Kind.Arg)
        return self.sym_tbl.add_symbol(symbol, node)

    def visit_BinaryOp(self, node: ir.BinaryOp):
        node = super().visit_BinaryOp(node)

        #node.left.add_user(node)
        #node.right.add_user(node)
        return node

    def visit_UnaryOp(self, node: ir.UnaryOp):
        node = super().visit_UnaryOp(node)
        #node.value.add_user(node)
        return node

    def visit_CallParam(self, node: ir.CallParam):
        node = super().visit_CallParam(node)
        #node.value.add_user(node)
        return node




# TODO: Unify the Constants to one single node for each value
