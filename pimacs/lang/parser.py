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

import pimacs.lang.ir as ir
import pimacs.lang.type as _type
from pimacs.lang.context import ModuleContext, Scope, Symbol, SymbolTable
from pimacs.lang.ir_visitor import IRMutator, IRVisitor
from pimacs.lang.sema import Sema, catch_sema_error


def get_lark_parser():
    dsl_grammar = open(os.path.join(os.path.dirname(__file__), "grammar.g")).read()
    return Lark(dsl_grammar, parser="lalr", postlex=PythonIndenter())


def get_parser(code: Optional[str] = None, filename: str = "<pimacs>"):
    source: ir.PlainCode | ir.FileName = (
        ir.PlainCode(code) if code else ir.FileName(filename)
    )
    dsl_grammar = open(os.path.join(os.path.dirname(__file__), "grammar.g")).read()
    return Lark(
        dsl_grammar,
        parser="lalr",
        postlex=PythonIndenter(),
        transformer=PimacsTransformer(source=source),
    )


def parse(
    code: str | None = None, filename: str = "<pimacs>", sema=False
) -> ir.File | None:
    if code:
        source = ir.PlainCode(code)
        parser = get_parser(code=code)
    else:
        code = open(filename).read()
        source = ir.FileName(filename)  # type: ignore
        parser = get_parser(code=None, filename=filename)

    stmts = parser.parse(code)
    the_ir = ir.File(stmts=stmts, loc=ir.Location(source, 0, 0))

    if sema:
        sema = Sema()
        the_ir = sema.visit(the_ir)
        if sema.succeed:
            return the_ir
        return None
    return the_ir


class PimacsTransformer(Transformer):
    def __init__(self, source: ir.PlainCode | ir.FileName):
        self.source = source

    def file_input(self, items):
        self._force_non_rule(items)
        return items

    def func_def(self, items):
        self._force_non_rule(items)
        name: str = items[0].value
        args: List[ir.ArgDecl] = safe_get(items, 1, [])
        type = safe_get(items, 2, None)
        block: ir.Block = safe_get(items, 3, None)
        loc = ir.Location(self.source, items[0].line, items[0].column)

        if block is None:
            raise Exception(f"{loc}:\nFunction {name} must have a block")

        return ir.FuncDecl(
            name=name, args=args or [], body=block, return_type=type, loc=loc
        )

    def func_args(self, items):
        self._force_non_rule(items)
        return items

    def func_arg(self, items) -> ir.ArgDecl:
        self._force_non_rule(items)
        name_token = safe_get(items, 0, None)
        name = name_token.value if name_token else None
        type_ = safe_get(items, 1, None)
        default = safe_get(items, 2, None)

        return ir.ArgDecl(
            name=name,
            type=type_,
            default=default,
            loc=ir.Location(self.source, items[0].line, items[0].column),
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

    def block(self, items) -> ir.Block:
        self._force_non_rule(items)
        stmts = list(filter(lambda x: x is not None, items))
        doc_string = None
        if isinstance(items[0], ir.DocString):
            doc_string = stmts[0]
            stmts = stmts[1:]

        return ir.Block(stmts=stmts, doc_string=doc_string, loc=stmts[0].loc)

    def return_stmt(self, items) -> ir.ReturnStmt:
        self._force_non_rule(items)
        assert len(items) == 2
        loc = ir.Location(self.source, items[0].line, items[0].column)
        if items[1]:
            return ir.ReturnStmt(value=items[1], loc=loc)
        return ir.ReturnStmt(value=None, loc=loc)

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
        type = items[0]
        spec_types = items[1]
        return items

    def func_call(self, items) -> ir.FuncCall | ir.LispFuncCall:
        self._force_non_rule(items)
        if isinstance(items[0], ir.VarRef):
            name = items[0].name
            loc = items[0].loc
        else:
            name = items[0].value
            loc = ir.Location(self.source, items[0].line, items[0].column)

        if len(items) == 2:
            args = items[1]
            type_spec = []
        elif len(items) == 3:
            type_spec = items[1]
            args = items[2]

        args = args if args else []

        if name.startswith("%"):
            assert loc
            func = ir.LispVarRef(name=name, loc=loc)
            assert not type_spec
            return ir.LispFuncCall(func=func, args=args, loc=loc)
        else:
            return ir.FuncCall(func=name, args=args, loc=loc, type_spec=type_spec)

    def lisp_symbol(self, items):
        self._force_non_rule(items)
        return ir.LispVarRef(
            name=items[0].value,
            loc=ir.Location(self.source, items[0].line, items[0].column),
        )

    def func_call_name(self, items) -> str:
        self._force_non_rule(items)
        return items[0]

    def call_param(self, items: List[lark.Tree]) -> ir.CallParam:
        self._force_non_rule(items)
        assert False
        assert len(items) == 1
        content = items[0][0]
        return ir.CallParam(name="", value=content, loc=content.loc)

    def call_param_name(self, items) -> Token:
        return items[0]

    def call_params(self, items: List[lark.Tree]) -> List[ir.CallParam]:
        self._force_non_rule(items)
        for item in items:
            assert isinstance(item, ir.CallParam)
        return items  # type: ignore

    def string(self, items: List[Token]):
        self._force_non_rule(items)
        loc = ir.Location(self.source, items[0].line, items[0].column)  # type: ignore
        return ir.Constant(value=items[0].value, loc=loc)

    def variable(self, items) -> ir.UnresolvedVarRef | ir.LispVarRef:
        self._force_non_rule(items)
        if isinstance(items[0], (ir.LispVarRef, ir.UnresolvedVarRef)):
            return items[0]
        loc = ir.Location(self.source, items[0].line, items[0].column)
        assert len(items) == 1
        if isinstance(items[0], ir.UnresolvedVarRef):
            return items[0]
        name = items[0].value
        if name.startswith("%"):
            return ir.LispVarRef(name=name, loc=loc)
        return ir.UnresolvedVarRef(name=items[0].value, loc=loc)

    def var_decl(self, items) -> ir.VarDecl:
        # self._force_non_rule(items)
        loc = ir.Location(self.source, items[0].line, items[0].column)
        name = items[1].value
        type = safe_get(items, 2, None)
        init = safe_get(items, 3, None)
        node = ir.VarDecl(name=name, type=type, init=init, loc=loc)
        # self._deduce_var_type(node)
        return node

    def let_decl(self, items) -> ir.VarDecl:
        self._force_non_rule(items)
        loc = ir.Location(self.source, items[0].line, items[0].column)
        name = items[1].value
        type = safe_get(items, 2, None)
        init = safe_get(items, 3, None)

        node = ir.VarDecl(name=name, type=type, init=init, loc=loc, mutable=False)
        return node

    def _deduce_var_type(self, var: ir.VarDecl) -> ir.VarDecl:
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
                    raise ValueError(
                        f"{var.loc}\nUnknown constant type {var.init.value}"
                    )
            else:
                var.type = var.init.type  # type: ignore

        return var

    def value_param(self, items):
        self._force_non_rule(items)
        assert len(items) == 1
        return ir.CallParam(name="", value=items[0], loc=items[0].loc)

    def key_value_param(self, items):
        self._force_non_rule(items)
        assert isinstance(items[0], Token)
        name = items[0].value
        value = items[1]
        loc = ir.Location(self.source, items[0].line, items[0].column)
        return ir.CallParam(name=name, value=value, loc=loc)

    def expr(self, items) -> lark.Tree:
        # self._force_non_rule(items)
        return items[0]

    def number(self, items: List[ir.Constant]):
        self._force_non_rule(items)
        assert len(items) == 1
        return items[0]

    def NUMBER(self, x):
        self._force_non_rule(x)
        value = float(x) if "." in x else int(x)
        return ir.Constant(value=value, loc=ir.Location(self.source, x.line, x.column))

    def atom(self, items):
        self._force_non_rule(items)
        return items[0]

    def true(self, items):
        self._force_non_rule(items)
        return ir.Constant(value=True, loc=None)

    def false(self, items):
        self._force_non_rule(items)
        return ir.Constant(value=False, loc=None)

    def nil(self, items):
        self._force_non_rule(items)
        return ir.Constant(value=None, loc=None)

    def if_stmt(self, items):
        self._force_non_rule(items)
        loc = ir.Location(self.source, items[0].line, items[0].column)
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

        return ir.IfStmt(
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
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.ADD, loc=items[0].loc
        )

    def sub(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.SUB, loc=items[0].loc
        )

    def mul(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.MUL, loc=items[0].loc
        )

    def div(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.DIV, loc=items[0].loc
        )

    def le(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.LE, loc=items[0].loc
        )

    def lt(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.LT, loc=items[0].loc
        )

    def ge(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.GE, loc=items[0].loc
        )

    def gt(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.GT, loc=items[0].loc
        )

    def eq(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.EQ, loc=items[0].loc
        )

    def ne(self, items):
        self._force_non_rule(items)
        return ir.BinaryOp(
            left=items[0], right=items[1], op=ir.BinaryOperator.NE, loc=items[0].loc
        )

    def decorator(self, items):
        self._force_non_rule(items)
        action = None
        args = []
        if isinstance(items[0], ir.FuncCall):
            action: ir.FuncCall = items[0]
            loc = action.loc
        elif isinstance(items[0], Token):
            if items[0].value == "template":
                loc = ir.Location(self.source, items[0].line, items[0].column)
                action = ir.Template(items[1])
            else:
                assert isinstance(items[0].value, str)
                action: str = items[0].value
                loc = ir.Location(self.source, items[0].line, items[0].column)

        return ir.Decorator(action=action, loc=loc)

    def decorated(self, items):
        self._force_non_rule(items)
        decorators: List[ir.Decorator] = items[:-1]
        func_def: ir.FuncDecl = items[-1]
        func_def.decorators = decorators
        return func_def

    def dotted_name(self, items) -> ir.UnresolvedVarRef:
        self._force_non_rule(items)
        return ir.UnresolvedVarRef(
            name=".".join([x.value for x in items]),
            loc=ir.Location(self.source, items[0].line, items[0].column),
        )

    def assign_stmt(self, items):
        self._force_non_rule(items)
        target = items[0]
        value = items[1]
        return ir.AssignStmt(target=target, value=value, loc=target.loc)

    def class_def(self, items):
        self._force_non_rule(items)
        name = items[0].value
        body = items[1]
        loc = ir.Location(self.source, items[0].line, items[0].column)
        return ir.ClassDef(name=name, body=body, loc=loc)

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
        return ir.DocString(
            content=content.strip(),
            loc=ir.Location(self.source, token.line, token.column),
        )

    def select_expr(self, items):
        self._force_non_rule(items)
        true_expr = items[0]
        cond = items[1]
        false_expr = items[2]

        assert isinstance(cond, ir.Expr)
        assert isinstance(true_expr, ir.Expr), f"{true_expr}"
        assert isinstance(false_expr, ir.Expr), f"{false_expr}"
        return ir.SelectExpr(
            cond=cond, then_expr=true_expr, else_expr=false_expr, loc=cond.loc
        )

    def guard_stmt(self, items):
        self._force_non_rule(items)
        func_call = items[0]
        body = items[1]

        return ir.GuardStmt(header=func_call, body=body, loc=func_call.loc)

    def not_cond(self, items):
        self._force_non_rule(items)
        loc = ir.Location(self.source, items[0].line, items[0].column)
        assert isinstance(items[1], ir.Expr)
        return ir.UnaryOp(op=ir.UnaryOperator.NOT, value=items[1], loc=loc)

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
