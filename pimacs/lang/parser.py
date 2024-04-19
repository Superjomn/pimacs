import os
from typing import List

import lark
from lark import Lark, Transformer
from lark.indenter import PythonIndenter

import pimacs.lang.ir as ir
import pimacs.lang.type as _type


def get_lark_parser():
    dsl_grammar = open(os.path.join(
        os.path.dirname(__file__), 'grammar.g')).read()
    return Lark(dsl_grammar, parser='lalr', postlex=PythonIndenter())


def get_parser(fllename: str = "<pimacs>"):
    dsl_grammar = open(os.path.join(
        os.path.dirname(__file__), 'grammar.g')).read()
    return Lark(dsl_grammar, parser='lalr', postlex=PythonIndenter(), transformer=PimacsTransformer(filename=fllename))


def parse(code: str, filename: str = "<pimacs>"):
    parser = get_parser(filename)
    stmts = parser.parse(code)
    return ir.File(body=stmts, loc=ir.Location(ir.FileName(filename), 0, 0))


class PimacsTransformer(Transformer):
    def __init__(self, filename: str):
        self.filename = ir.FileName(filename)

    def file_input(self, items):
        return items

    def func_def(self, items):
        name: str = items[0].value
        args: List[ir.ArgDecl] = safe_get(items, 1, [])
        type = safe_get(items, 2, None)
        block: ir.Block = safe_get(items, 3, None)
        loc = ir.Location(self.filename, items[0].line, items[0].column)

        if block is None:
            raise Exception(f"{loc}:\nFunction {name} must have a block")

        return ir.FuncDecl(name=name, args=args, body=block,
                           return_type=type,
                           loc=loc)

    def func_args(self, items):
        return items

    def func_arg(self, items) -> ir.ArgDecl:
        name: str = safe_get(items, 0, None)
        type_ = safe_get(items, 1, None)
        default = safe_get(items, 2, None)

        return ir.ArgDecl(name=name, type=type_, default=default, loc=ir.Location(self.filename, items[0].line, items[0].column))

    def func_params(self, items):
        return items[0]

    def func_param(self, items):
        name: str = items[0].value
        type = safe_get(items, 1, None)
        default = safe_get(items, 2, None)

        if type is None:
            type = _type.Nil

        loc = ir.Location(self.filename, items[0].line, items[0].column)

        return ir.ArgDecl(name=name, type=type, default=default, loc=loc)

    def type(self, items):
        return items[0]

    def PRIMITIVE_TYPE(self, items) -> _type.Type:
        # This terminal is failed, because it is covered by the custom_type
        raise NotImplementedError()

    def custom_type(self, items) -> _type.Type:
        assert len(items) == 1
        ret = _type.parse_primitive_type(items[0].value)
        if ret is not None:
            return ret
        return _type.Type(type_id=_type.TypeId.CUSTOMED, _name=items[0].value)

    def block(self, items) -> ir.Block:
        stmts = list(filter(lambda x: x is not None, items))

        return ir.Block(stmts=stmts, loc=stmts[0].loc)

    def assign_stmt(self, items) -> ir.AssignStmt:
        return items

    def return_stmt(self, items) -> ir.ReturnStmt:
        return ir.ReturnStmt(value=items[0], loc=items[0].loc)

    def func_call(self, items) -> ir.FuncDecl:
        name = items[0].value
        args = items[1]
        loc = ir.Location(self.filename, items[0].line, items[0].column)
        return ir.FuncCall(func=name, args=args, loc=loc)

    def func_call_name(self, items) -> str:
        return items[0]

    def call_param(self, items: List[lark.Tree]) -> ir.CallParam:
        assert len(items) == 1
        content = items[0][0]
        return ir.CallParam(name="", value=content, loc=content.loc)

    def call_params(self, items: List[lark.Tree]) -> List[ir.CallParam]:
        return items[0]

    def string(self, items: List[lark.lexer.Token]):
        loc = ir.Location("", items[0].line, items[0].column)
        return ir.Constant(value=items[0].value, loc=loc)

    def variable(self, items) -> ir.VarRef:
        assert len(items) == 1
        if isinstance(items[0], ir.VarRef):
            return items[0]
        name = items[0].value
        if name.startswith('%'):
            return ir.LispVarRef(name=name, loc=ir.Location(self.filename, items[0].line, items[0].column))
        return ir.VarRef(name=items[0].value, loc=ir.Location(self.filename, items[0].line, items[0].column))

    def var_decl(self, items) -> ir.VarDecl:
        name = items[0].value
        type = safe_get(items, 1, None)
        init = safe_get(items, 2, None)
        loc = ir.Location(self.filename, items[0].line, items[0].column)
        return ir.VarDecl(name=name, type=type, init=init, loc=loc)

    def value_param(self, items):
        return items

    def expr(self, items):
        return items[0]

    def number(self, x: List[ir.Constant]):
        assert len(x) == 1
        return x[0]

    def NUMBER(self, x):
        value = float(x) if '.' in x else int(x)
        return ir.Constant(value=value, loc=ir.Location(self.filename, x.line, x.column))

    def atom(self, items):
        return items[0]

    def true(self, items):
        return ir.Constant(value=True, loc=None)

    def false(self, items):
        return ir.Constant(value=False, loc=None)

    def nil(self, items):
        return ir.Constant(value=None, loc=None)

    def if_stmt(self, items):
        cond = items[0]
        block = items[1]
        return ir.IfStmt(cond=cond, then_branch=block, loc=cond.loc)

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
            loc = ir.Location(self.filename, items[0].line, items[0].column)

        return ir.Decorator(action=action, loc=loc)

    def decorated(self, items):
        decorators: List[ir.Decorator] = items[:-1]
        func_def: ir.FuncDecl = items[-1]
        func_def.decorators = decorators
        return func_def

    def dotted_name(self, items) -> ir.VarRef:
        return ir.VarRef(name=".".join([x.value for x in items]), loc=ir.Location(self.filename, items[0].line, items[0].column))

    def assign_stmt(self, items):
        target = items[0]
        value = items[1]
        return ir.AssignStmt(target=target, value=value, loc=target.loc)

    def class_def(self, items):
        name = items[0].value
        body = items[1]
        loc = ir.Location(self.filename, items[0].line, items[0].column)
        return ir.ClassDef(name=name, body=body, loc=loc)

    def class_body(self, items):
        return filter(lambda x: x is not None, items)


def safe_get(items, index, default):
    if len(items) > index:
        return items[index]
    return default
