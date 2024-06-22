import os
from typing import *

import lark
from lark import Lark, Transformer
from lark.indenter import PythonIndenter
from lark.lexer import Token

import pimacs.ast.ast as ast
import pimacs.ast.type as ty


def get_lark_parser():
    dsl_grammar = open(os.path.join(
        os.path.dirname(__file__), "grammar.g")).read()
    return Lark(dsl_grammar, parser="lalr", postlex=PythonIndenter())


def get_parser(code: Optional[str] = None, filename: str = "<pimacs>"):
    source: ast.PlainCode | ast.FileName = (
        ast.PlainCode(code) if code else ast.FileName(filename)
    )
    dsl_grammar = open(os.path.join(
        os.path.dirname(__file__), "grammar.g")).read()
    return Lark(
        dsl_grammar,
        parser="lalr",
        postlex=PythonIndenter(),
        transformer=PimacsTransformer(source=source),
        propagate_positions=True,
    )


class PimacsTransformer(Transformer):
    ''' This class is used to transform the AST from Lark to Pimacs AST.
    It only perform the transformation, and does not perform any semantic analysis, so lots of
    unsolved nodes are left in the AST.
    '''

    def __init__(self, source: ast.PlainCode | ast.FileName):
        self.source = source

    def file_input(self, items):
        self._force_non_rule(items)
        return items

    def _get_loc(self, token: Token) -> ast.Location:
        assert token.line is not None
        assert token.column is not None
        return ast.Location(self.source, token.line, token.column)

    def func_def(self, items):
        self._force_non_rule(items)
        template_params = tuple()
        if len(items) == 4:
            name: str = items[0].value
            args: List[ast.Arg] = safe_get(items, 1, tuple())
            type = safe_get(items, 2, None)
            block: ast.Block = safe_get(items, 3, None)
        elif len(items) == 5:
            name: str = items[0].value
            template_params = items[1]
            args: List[ast.Arg] = safe_get(items, 2, tuple())
            type = safe_get(items, 3, None)
            block: ast.Block = safe_get(items, 4, None)
        else:
            raise ValueError(f"Unknown function def {items}")

        loc = self._get_loc(items[0])

        if block is None:
            raise Exception(f"{loc}:\nFunction {name} must have a block")
        if isinstance(args, list):
            args = tuple(args)

        return ast.Function(
            name=name, args=args or tuple(), body=block, return_type=type, loc=loc,
            template_params=template_params,
        )

    def func_args(self, items):
        self._force_non_rule(items)
        return items

    def func_arg(self, items) -> ast.Arg:
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

        return ast.Arg(
            name=name,
            type=type_,
            default=default,
            loc=self._get_loc(items[0]),
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
                    items[0] = items[0].get_optional_type()
            return items[0]
        else:
            raise ValueError(f"Unknown type {items}")

    def set_type(self, items):
        "The type of Set"
        self._force_non_rule(items)
        assert len(items) == 1
        return ty.Set_.clone_with(*items)

    def dict_type(self, items):
        """
        Type of Dict.
        """
        self._force_non_rule(items)
        assert len(items) == 2
        return ty.Dict_.clone_with(*items)

    def list_type(self, items):
        """Type of List."""
        self._force_non_rule(items)
        assert len(items) == 1
        return ty.List_.clone_with(*items)

    def PRIMITIVE_TYPE(self, items) -> ty.Type:
        self._force_non_rule(items)
        # This terminal is failed, because it is covered by the custom_type
        raise NotImplementedError()

    def custom_type(self, items) -> ty.Type:
        self._force_non_rule(items)
        assert len(items) == 1
        ret = ty.Type.get_primitive(items[0].value)
        if ret is not None:
            return ret
        return ty.GenericType(name=items[0].value)

    def block(self, items) -> ast.Block:
        self._force_non_rule(items)
        stmts = list(filter(lambda x: x is not None, items))
        doc_string = None
        if isinstance(items[0], ast.DocString):
            doc_string = stmts[0]
            stmts = stmts[1:]

        return ast.Block(stmts=tuple(stmts), doc_string=doc_string, loc=stmts[0].loc)

    def return_stmt(self, items) -> ast.Return:
        self._force_non_rule(items)
        assert len(items) == 2
        loc = self._get_loc(items[0])
        if items[1]:
            return ast.Return(value=items[1], loc=loc)
        return ast.Return(value=None, loc=loc)

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

        if len(spec_types) == 1 and spec_types[0].is_List():
            spec_types = spec_types[0].inner_types
        if isinstance(spec_types, list):
            spec_types = tuple(spec_types)

        # TODO: Unify the types in ModuleContext
        return ty.CompositeType(name=type.value, params=spec_types)

    def func_call(self, items) -> ast.Call | ast.LispCall:
        self._force_non_rule(items)
        if isinstance(items[0], ast.VarRef):
            name = items[0].name
            loc = items[0].loc
        elif isinstance(items[0], ast.UVarRef):
            name = items[0].name
            loc = items[0].loc
        elif isinstance(items[0], ast.UAttr):
            args = list(filter(lambda x: x is not None, items[1:]))
            if args and isinstance(args[0], list):
                args = args[0]
            return ast.Call(target=items[0], args=tuple(args), loc=items[0].loc)
        else:
            name = items[0].value
            loc = self._get_loc(items[0])

        if len(items) == 2:
            args = items[1]
            type_spec = []
        elif len(items) == 3:
            type_spec = items[1]
            args = items[2]

        args = args if args else []

        if name.startswith('%'):
            return ast.LispCall(target=name[1:], args=tuple(args), loc=loc)

        the_func = ast.UFunction(name=name, loc=loc)
        return ast.Call(target=the_func, args=tuple(args), loc=loc, type_spec=tuple(type_spec))

    def lisp_symbol(self, items):
        self._force_non_rule(items)
        return ast.UVarRef(
            # TODO: Represent the lisp symbols in a better way
            name='%'+items[0].value,
            loc=self._get_loc(items[0]),
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
        loc = self._get_loc(items[0])
        return ast.Literal(value=items[0].value, loc=loc)

    def variable(self, items):
        self._force_non_rule(items)
        if isinstance(items[0], (ast.UVarRef, ast.UAttr)):
            return items[0]

        loc = self._get_loc(items[0])
        name = items[0].value
        return ast.UVarRef(name=name, loc=loc)

    def var_decl(self, items) -> ast.VarDecl:
        # self._force_non_rule(items)
        loc = self._get_loc(items[0])
        name = items[1].value
        type = safe_get(items, 2, ty.Unk)
        init = safe_get(items, 3, None)
        if isinstance(init, lark.Tree):
            init = init.children[0]
        if init:
            assert isinstance(init, ast.Node), f"get {init}"
        node = ast.VarDecl(name=name, type=type, init=init, loc=loc)
        return node

    def let_decl(self, items) -> ast.VarDecl:
        self._force_non_rule(items)
        loc = self._get_loc(items[0])
        name = items[1].value
        type = safe_get(items, 2, ty.Unk)
        init = safe_get(items, 3, None)

        node = ast.VarDecl(name=name, type=type, init=init,
                           loc=loc, mutable=False)
        return node

    def value_param(self, items):
        self._force_non_rule(items)
        assert len(items) == 1
        return ast.CallParam(name="", value=items[0], loc=items[0].loc)

    def key_value_param(self, items):
        self._force_non_rule(items)
        assert isinstance(items[0], Token)
        name = items[0].value
        value = items[1]
        loc = self._get_loc(items[0])
        return ast.CallParam(name=name, value=value, loc=loc)

    def expr(self, items) -> lark.Tree:
        # self._force_non_rule(items)
        return items[0]

    def number(self, items: List[ast.Literal]):
        self._force_non_rule(items)
        assert len(items) == 1
        return items[0]

    def NUMBER(self, x):
        self._force_non_rule(x)
        value = float(x) if "." in x else int(x)
        return ast.Literal(value=value, loc=self._get_loc(x))

    def atom(self, items):
        self._force_non_rule(items)
        return items[0]

    def true(self, items):
        self._force_non_rule(items)
        return ast.Literal(value=True, loc=None)

    def false(self, items):
        self._force_non_rule(items)
        return ast.Literal(value=False, loc=None)

    def nil(self, items):
        self._force_non_rule(items)
        return ast.Literal(value=None, loc=None)

    def if_stmt(self, items):
        self._force_non_rule(items)
        loc = self._get_loc(items[0])
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

        return ast.If(
            cond=cond,
            then_branch=then_block,
            elif_branches=tuple(elif_blocks),
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
        if isinstance(items[0], ast.Call):
            action: ast.Call = items[0]
            loc = action.loc
        elif isinstance(items[0], Token):
            loc = self._get_loc(items[0])
            if items[0].value == "template":
                # convert GenericType to PlaceholderType
                placeholders = tuple(
                    list(ty.PlaceholderType(gty.name) for gty in items[1]))
                action = ast.Template(placeholders)
            else:
                assert isinstance(items[0].value, str)
                action: str = items[0].value

        return ast.Decorator(action=action, loc=loc)

    def decorated(self, items):
        self._force_non_rule(items)
        decorators: List[ast.Decorator] = items[:-1]
        innder: ast.Function | ast.Class | ast.VarDecl = items[-1]
        innder.decorators = decorators if isinstance(
            decorators, tuple) else tuple(decorators)
        return innder

    def dotted_name(self, items):
        """ Get attribute. """
        self._force_non_rule(items)
        assert len(items) > 1

        attr = None
        for i, token in enumerate(items[:-1]):
            name = token.value
            value = attr or ast.UVarRef(name=name, loc=self._get_loc(token))
            attr = ast.UAttr(
                value=value, attr=items[i+1].value, loc=self._get_loc(token))

        return attr

    def assign_stmt(self, items):
        self._force_non_rule(items)
        target = items[0]
        value = items[1]

        if isinstance(target, Token):
            target = ast.UVarRef(name=target.value, loc=self._get_loc(target))

        return ast.Assign(target=target, value=value, loc=target.loc)

    def class_def(self, items):
        self._force_non_rule(items)
        template_params = tuple()
        if len(items) == 2:
            name = items[0].value
            body = items[1]
        elif len(items) == 3:
            name = items[0].value
            template_params = items[1]
            body = items[2]
        else:
            raise ValueError(f"Unknown class def {items}")

        loc = self._get_loc(items[0])
        return ast.Class(name=name, body=body,
                         template_params=template_params,
                         loc=loc)

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
            loc=self._get_loc(token),
        )

    def select_expr(self, items):
        self._force_non_rule(items)
        true_expr = items[0]
        cond = items[1]
        false_expr = items[2]

        assert isinstance(cond, ast.Expr)
        assert isinstance(true_expr, ast.Expr), f"{true_expr}"
        assert isinstance(false_expr, ast.Expr), f"{false_expr}"
        return ast.Select(
            cond=cond, then_expr=true_expr, else_expr=false_expr, loc=cond.loc
        )

    def guard_stmt(self, items):
        self._force_non_rule(items)
        func_call = items[0]
        body = items[1]

        return ast.Guard(header=func_call, body=body, loc=func_call.loc)

    def not_cond(self, items):
        self._force_non_rule(items)
        loc = self._get_loc(items[0])
        assert isinstance(items[1], ast.Expr)
        return ast.UnaryOp(op=ast.UnaryOperator.NOT, operand=items[1], loc=loc)

    def type_placeholders(self, items):
        self._force_non_rule(items)
        return tuple([ty.PlaceholderType(name=item.value) for item in items])

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
