'''
This file contains the AST nodes used in the LISP codegen.
'''
from dataclasses import dataclass, field
from typing import List as _List
from typing import Optional

import pimacs.ast.ast as ast


@dataclass(slots=True)
class Node:
    ''' Node is the base class for all AST nodes. '''
    loc: Optional[ast.Location] = field(
        compare=False, repr=False, init=False, default=None)


@dataclass(slots=True)
class Literal(Node):
    ''' Literal is the base class for all literals. '''
    value: int | float | str | bool


@dataclass(slots=True)
class Expr(Node):
    pass


class VarDecl(Node):
    name: str
    default: Optional[Expr] = None


@dataclass(slots=True)
class Symbol(Node):
    ''' Symbols such as 'add, 'defun and so on. '''
    name: str


@dataclass(slots=True)
class List(Expr):
    ''' List expression. '''
    elements: _List[Expr]


@dataclass(slots=True)
class Let(Expr):
    ''' Let expression. '''
    vars: _List[VarDecl]
    body: _List[Expr]


@dataclass(slots=True)
class Guard(Expr):
    header: str
    args: _List[VarDecl]
    body: _List[Expr]


@dataclass(slots=True)
class Function(Node):
    name: str
    args: _List[VarDecl]
    body: _List[Expr]


@dataclass(slots=True)
class Call(Expr):
    func: str
    args: _List[Expr]


@dataclass(slots=True)
class If(Expr):
    cond: VarDecl | Expr
    then_block: Let
    else_block: Let


@dataclass(slots=True)
class While(Expr):
    cond: VarDecl | Expr
    body: Let
