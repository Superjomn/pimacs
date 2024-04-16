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


class PimacsTransformer(Transformer):
    def __init__(self, filename: str):
        self.filename = ir.FileName(filename)

    def file_input(self, items):
        return items

    def func_def(self, items):
        name: str = items[0].value
        args: List[ir.ArgDecl] = safe_get(items, 1, [])
        block: ir.Block = safe_get(items, 3, None)
        loc = ir.Location(self.filename, items[0].line, items[0].column)

        if block is None:
            raise Exception(f"{loc}:\nFunction {name} must have a block")

        return ir.FuncDecl(name=name, args=args, body=block, loc=loc)

    def func_params(self, items):
        return items

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

    def func_call(self, items) -> ir.FuncDecl:
        name = items[0].value
        args = items[1]
        loc = ir.Location(self.filename, items[0].line, items[0].column)
        return ir.FuncCall(func=name, args=args, loc=loc)

    def block(self, items) -> ir.Block:
        stmts = list(filter(lambda x: x is not None, items))
        return ir.Block(stmts=stmts, loc=stmts[0].loc)

    def args(self, items: List[lark.lexer.Token]) -> List[ir.CallParam]:
        return items

    def arg(self, items: List[lark.Tree]) -> ir.CallParam:
        assert len(items) == 1
        content = items[0][0]
        return ir.CallParam(name="", value=content, loc=content.loc)

    def string(self, items: List[lark.lexer.Token]):
        loc = ir.Location("", items[0].line, items[0].column)
        return ir.Constant(value=items[0].value, loc=loc)

    def var_decl(self, items) -> ir.VarDecl:
        name = items[0].value
        type = safe_get(items, 1, None)
        init = safe_get(items, 2, None)
        loc = ir.Location(self.filename, items[0].line, items[0].column)
        if init:
            assert len(init) == 1
            init = init[0]
        return ir.VarDecl(name=name, type=type, init=init, loc=loc)

    def expr(self, items):
        return items

    def number(self, x: List[ir.Constant]):
        assert len(x) == 1
        return x[0]

    def NUMBER(self, x):
        value = float(x) if '.' in x else int(x)
        return ir.Constant(value=value, loc=ir.Location(self.filename, x.line, x.column))


def safe_get(items, index, default):
    if len(items) > index:
        return items[index]
    return default
