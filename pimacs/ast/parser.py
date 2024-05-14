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
from lark.lexer import Token

import pimacs.ast.ast as ast
import pimacs.ast.type as _type
from pimacs.sema.ast_visitor import IRMutator, IRVisitor
from pimacs.sema.context import ModuleContext, Scope, Symbol, SymbolTable
from pimacs.sema.sema import Sema, catch_sema_error


def get_lark_parser():
    dsl_grammar = open(os.path.join(os.path.dirname(__file__), "grammar.g")).read()
    return Lark(dsl_grammar, parser="lalr", postlex=PythonIndenter())


def get_parser(code: Optional[str] = None, filename: str = "<pimacs>"):
    source: ast.PlainCode | ast.FileName = (
        ast.PlainCode(code) if code else ast.FileName(filename)
    )
    dsl_grammar = open(os.path.join(os.path.dirname(__file__), "grammar.g")).read()
    return Lark(
        dsl_grammar,
        parser="lalr",
        postlex=PythonIndenter(),
        transformer=PimacsTransformer(source=source),
    )


class PimacsTransformer(Transformer):
    def __init__(self, source: ast.PlainCode | ast.FileName):
        self.source = source
        self.ctx = ModuleContext("unk")

    def file_input(self, items):
        self._force_non_rule(items)
        return items

    def func_def(self, items):
        self._force_non_rule(items)
        name: str = items[0].value
        args: List[ast.ArgDecl] = safe_get(items, 1, [])
        type = safe_get(items, 2, None)
        block: ast.Block = safe_get(items, 3, None)
        loc = ast.Location(self.source, items[0].line, items[0].column)

        if block is None:
            raise Exception(f"{loc}:\nFunction {name} must have a block")

        return ast.FuncDecl(
            name=name, args=args or [], body=block, return_type=type, loc=loc
        )

    def func_args(self, items):
        self._force_non_rule(items)
        return items

    def func_arg(self, items) -> ast.ArgDecl:
        self._force_non_rule(items)
        name_token = safe_get(items, 0, None)
        name = name_token.value if name_token else None
        type_ = safe_get(items, 1, None)
        is_variadic = False
        if isinstance(type_, tuple):
            assert type_[1] == "variaric"
            is_variadic = True
            type_ = type_[0]

        default = safe_get(items, 2, None)

        return ast.ArgDecl(
            name=name,
            type=type_,
            default=default,
            loc=ast.Location(self.source, items[0].line, items[0].column),
            is_variadic=is_variadic,
        )

    def type_base(self, items):
        self._force_non_rule(items)
        return items[0]

    def type(self, items):
        self._force_non_rule(items)
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            if items[1]:
                if items[1].value == "?":
                    items[0].is_optional = True
            return items[0]
        else:
            raise ValueError(f"Unknown type {items}")

    def set_type(self, items):
        "The type of Set"
        self._force_non_rule(items)
        assert len(items) == 1
        return _type.SetType(inner_types=items)

    def dict_type(self, items):
        """
        Type of Dict.
        """
        self._force_non_rule(items)
        assert len(items) == 2
        return _type.DictType(key_type=items[0], value_type=items[1])

    def list_type(self, items):
        """Type of List."""
        self._force_non_rule(items)
        assert len(items) == 1
        return _type.ListType(inner_types=items)

    def PRIMITIVE_TYPE(self, items) -> _type.Type:
        self._force_non_rule(items)
        # This terminal is failed, because it is covered by the custom_type
        raise NotImplementedError()

    def custom_type(self, items) -> _type.Type:
        self._force_non_rule(items)
        assert len(items) == 1
        ret = _type.parse_primitive_type(items[0].value)
        if ret is not None:
            return ret
        return _type.Type(type_id=_type.TypeId.CUSTOMED, name=items[0].value)

    def block(self, items) -> ast.Block:
        self._force_non_rule(items)
        stmts = list(filter(lambda x: x is not None, items))
        doc_string = None
        if isinstance(items[0], ast.DocString):
            doc_string = stmts[0]
            stmts = stmts[1:]

        return ast.Block(stmts=stmts, doc_string=doc_string, loc=stmts[0].loc)

    def return_stmt(self, items) -> ast.ReturnStmt:
        self._force_non_rule(items)
        assert len(items) == 2
        loc = ast.Location(self.source, items[0].line, items[0].column)
        if items[1]:
            return ast.ReturnStmt(value=items[1], loc=loc)
        return ast.ReturnStmt(value=None, loc=loc)

    def type_spec(self, items):
        self._force_non_rule(items)
        return items[0]

    def type_list(self, items):
        self._force_non_rule(items)
        return items

    def variadic_type(self, items):
        type = items[0]
        return type, "variaric"

    def complex_type(self, items):
        # a known bug, that the TT[T] could be interpreted as "TT" "[T]", here just restore it to a complex type
        type = items[0]
        spec_types = items[1]

        if len(spec_types) == 1 and spec_types[0].type_id is _type.TypeId.List:
            spec_types = spec_types[0].inner_types

        # TODO: Unify the types in ModuleContext
        return _type.make_customed(name=type.value, subtypes=spec_types)

    def func_call(self, items) -> ast.FuncCall | ast.LispFuncCall:
        self._force_non_rule(items)
        if isinstance(items[0], ast.VarRef):
            name = items[0].name
            loc = items[0].loc
        elif isinstance(items[0], ast.UnresolvedVarRef):
            name = items[0].name
            loc = items[0].loc
        else:
            name = items[0].value
            loc = ast.Location(self.source, items[0].line, items[0].column)

        if len(items) == 2:
            args = items[1]
            type_spec = []
        elif len(items) == 3:
            type_spec = items[1]
            args = items[2]

        args = args if args else []

        if name.startswith("%"):
            assert loc
            func = ast.LispVarRef(name=name, loc=loc)
            assert not type_spec
            return ast.LispFuncCall(func=func, args=args, loc=loc)
        else:
            return ast.FuncCall(func=name, args=args, loc=loc, type_spec=type_spec)

    def lisp_symbol(self, items):
        self._force_non_rule(items)
        return ast.LispVarRef(
            name=items[0].value,
            loc=ast.Location(self.source, items[0].line, items[0].column),
        )

    def func_call_name(self, items) -> str:
        self._force_non_rule(items)
        return items[0]

    def call_param(self, items: List[lark.Tree]) -> ast.CallParam:
        self._force_non_rule(items)
        assert False
        assert len(items) == 1
        content = items[0][0]
        return ast.CallParam(name="", value=content, loc=content.loc)

    def call_param_name(self, items) -> Token:
        return items[0]

    def call_params(self, items: List[lark.Tree]) -> List[ast.CallParam]:
        self._force_non_rule(items)
        for item in items:
            assert isinstance(item, ast.CallParam)
        return items  # type: ignore

    def string(self, items: List[Token]):
        self._force_non_rule(items)
        loc = ast.Location(self.source, items[0].line, items[0].column)  # type: ignore
        return ast.Constant(value=items[0].value, loc=loc)

    def variable(self, items) -> ast.UnresolvedVarRef | ast.LispVarRef:
        self._force_non_rule(items)
        if isinstance(items[0], (ast.LispVarRef, ast.UnresolvedVarRef)):
            return items[0]
        loc = ast.Location(self.source, items[0].line, items[0].column)
        assert len(items) == 1
        if isinstance(items[0], ast.UnresolvedVarRef):
            return items[0]
        name = items[0].value
        if name.startswith("%"):
            return ast.LispVarRef(name=name, loc=loc)
        return ast.UnresolvedVarRef(name=items[0].value, loc=loc)

    def var_decl(self, items) -> ast.VarDecl:
        # self._force_non_rule(items)
        loc = ast.Location(self.source, items[0].line, items[0].column)
        name = items[1].value
        type = safe_get(items, 2, None)
        init = safe_get(items, 3, None)
        node = ast.VarDecl(name=name, type=type, init=init, loc=loc)
        # self._deduce_var_type(node)
        return node

    def let_decl(self, items) -> ast.VarDecl:
        self._force_non_rule(items)
        loc = ast.Location(self.source, items[0].line, items[0].column)
        name = items[1].value
        type = safe_get(items, 2, None)
        init = safe_get(items, 3, None)

        node = ast.VarDecl(name=name, type=type, init=init, loc=loc, mutable=False)
        return node

    def _deduce_var_type(self, var: ast.VarDecl) -> ast.VarDecl:
        if var.type is not None:
            return var

        if var.init:
            if isinstance(var.init, ast.LispVarRef):
                var.type = _type.LispType
            elif isinstance(var.init, ast.Constant):
                if isinstance(var.init.value, int):
                    var.type = _type.Int
                elif isinstance(var.init.value, float):
                    var.type = _type.Float
                elif isinstance(var.init.value, str):
                    var.type = _type.Str
                elif var.init.value is None:
                    var.type = _type.Nil
                else:
                    raise ValueError(
                        f"{var.loc}\nUnknown constant type {var.init.value}"
                    )
            else:
                var.type = var.init.type  # type: ignore

        return var

    def value_param(self, items):
        self._force_non_rule(items)
        assert len(items) == 1
        return ast.CallParam(name="", value=items[0], loc=items[0].loc)

    def key_value_param(self, items):
        self._force_non_rule(items)
        assert isinstance(items[0], Token)
        name = items[0].value
        value = items[1]
        loc = ast.Location(self.source, items[0].line, items[0].column)
        return ast.CallParam(name=name, value=value, loc=loc)

    def expr(self, items) -> lark.Tree:
        # self._force_non_rule(items)
        return items[0]

    def number(self, items: List[ast.Constant]):
        self._force_non_rule(items)
        assert len(items) == 1
        return items[0]

    def NUMBER(self, x):
        self._force_non_rule(x)
        value = float(x) if "." in x else int(x)
        return ast.Constant(
            value=value, loc=ast.Location(self.source, x.line, x.column)
        )

    def atom(self, items):
        self._force_non_rule(items)
        return items[0]

    def true(self, items):
        self._force_non_rule(items)
        return ast.Constant(value=True, loc=None)

    def false(self, items):
        self._force_non_rule(items)
        return ast.Constant(value=False, loc=None)

    def nil(self, items):
        self._force_non_rule(items)
        return ast.Constant(value=None, loc=None)

    def if_stmt(self, items):
        self._force_non_rule(items)
        loc = ast.Location(self.source, items[0].line, items[0].column)
        cond = items[1]
        then_block = items[2]

        elif_blocks = []
        else_block = None

        for item in items[3:]:
            if item[0] == "elif":
                elif_blocks.append(item[1:])
            elif item[0] == "else":
                else_block = item[1]
            else:
                raise ValueError(f"{loc}\nUnknown if block {item}")

        return ast.IfStmt(
            cond=cond,
            then_branch=then_block,
            elif_branches=elif_blocks,
            else_branch=else_block,
            loc=loc,
        )

    def elif_block(self, items):
        return "elif", items[1], items[2]  # cond, block

    def else_block(self, items):
        return "else", items[1]  # block

    def add(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.ADD, loc=items[0].loc
        )

    def sub(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.SUB, loc=items[0].loc
        )

    def mul(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.MUL, loc=items[0].loc
        )

    def div(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.DIV, loc=items[0].loc
        )

    def le(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.LE, loc=items[0].loc
        )

    def lt(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.LT, loc=items[0].loc
        )

    def ge(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.GE, loc=items[0].loc
        )

    def gt(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.GT, loc=items[0].loc
        )

    def eq(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.EQ, loc=items[0].loc
        )

    def ne(self, items):
        self._force_non_rule(items)
        return ast.BinaryOp(
            left=items[0], right=items[1], op=ast.BinaryOperator.NE, loc=items[0].loc
        )

    def decorator(self, items):
        self._force_non_rule(items)
        action = None
        args = []
        if isinstance(items[0], ast.FuncCall):
            action: ast.FuncCall = items[0]
            loc = action.loc
        elif isinstance(items[0], Token):
            if items[0].value == "template":
                loc = ast.Location(self.source, items[0].line, items[0].column)
                action = ast.Template(items[1])
            else:
                assert isinstance(items[0].value, str)
                action: str = items[0].value
                loc = ast.Location(self.source, items[0].line, items[0].column)

        return ast.Decorator(action=action, loc=loc)

    def decorated(self, items):
        self._force_non_rule(items)
        decorators: List[ast.Decorator] = items[:-1]
        func_def: ast.FuncDecl = items[-1]
        func_def.decorators = decorators
        return func_def

    def dotted_name(self, items) -> ast.UnresolvedVarRef:
        self._force_non_rule(items)
        return ast.UnresolvedVarRef(
            name=".".join([x.value for x in items]),
            loc=ast.Location(self.source, items[0].line, items[0].column),
        )

    def assign_stmt(self, items):
        self._force_non_rule(items)
        target = items[0]
        value = items[1]
        return ast.AssignStmt(target=target, value=value, loc=target.loc)

    def class_def(self, items):
        self._force_non_rule(items)
        name = items[0].value
        body = items[1]
        loc = ast.Location(self.source, items[0].line, items[0].column)
        return ast.ClassDef(name=name, body=body, loc=loc)

    def class_body(self, items):
        self._force_non_rule(items)
        return filter(lambda x: x is not None, items)

    def STRING(self, items):
        self._force_non_rule(items)
        assert isinstance(items, str)
        return items

    def doc_string(self, items):
        self._force_non_rule(items)
        token = items[0]
        assert token.value.startswith('"') and token.value.endswith('"')
        content = token.value[1:-1]
        return ast.DocString(
            content=content.strip(),
            loc=ast.Location(self.source, token.line, token.column),
        )

    def select_expr(self, items):
        self._force_non_rule(items)
        true_expr = items[0]
        cond = items[1]
        false_expr = items[2]

        assert isinstance(cond, ast.Expr)
        assert isinstance(true_expr, ast.Expr), f"{true_expr}"
        assert isinstance(false_expr, ast.Expr), f"{false_expr}"
        return ast.SelectExpr(
            cond=cond, then_expr=true_expr, else_expr=false_expr, loc=cond.loc
        )

    def guard_stmt(self, items):
        self._force_non_rule(items)
        func_call = items[0]
        body = items[1]

        return ast.GuardStmt(header=func_call, body=body, loc=func_call.loc)

    def not_cond(self, items):
        self._force_non_rule(items)
        loc = ast.Location(self.source, items[0].line, items[0].column)
        assert isinstance(items[1], ast.Expr)
        return ast.UnaryOp(op=ast.UnaryOperator.NOT, value=items[1], loc=loc)

    def type_placeholders(self, items):
        self._force_non_rule(items)
        return [_type.make_customed(name=item.value) for item in items]

    def type_placeholder_list(self, items):
        self._force_non_rule(items)
        return items[0]

    def _force_non_rule(self, items):
        items = [items] if not isinstance(items, list) else items
        trees = list(filter(lambda x: isinstance(x, lark.Tree), items))
        assert not trees, f"Unknown rule {trees}"


def safe_get(items, index, default):
    if len(items) > index:
        return items[index]
    return default
