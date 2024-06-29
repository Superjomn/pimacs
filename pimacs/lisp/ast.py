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
class Module(Node):
    name: str
    stmts: _List[Node] = field(default_factory=list)


@dataclass(slots=True)
class Literal(Node):
    ''' Literal is the base class for all literals. '''
    value: int | float | str | bool


@dataclass(slots=True)
class Expr(Node):
    pass


@dataclass(slots=True)
class VarDecl(Node):
    name: str
    init: Optional[Expr] = None


@dataclass(slots=True)
class VarRef(Expr):
    ''' Reference to a variable. '''
    name: str


@dataclass(slots=True)
class Symbol(Node):
    ''' Symbols such as 'add, 'defun and so on. '''
    name: str


@dataclass(slots=True)
class List(Expr):
    ''' List expression. '''
    elements: _List[Expr | str]


@dataclass(slots=True)
class Let(Expr):
    ''' Let expression. '''
    vars: _List[VarDecl]
    body: _List[Expr]


@dataclass(slots=True)
class Guard(Expr):
    header: Expr
    body: _List[Expr]


@dataclass(slots=True)
class Return(Expr):
    ''' Return expression.
    This is a mapping for `cl-return` in LISP.'''
    block_name: str
    value: Expr | None = field(default=None)


@dataclass(slots=True)
class Assign(Expr):
    ''' Assignment expression. '''
    target: VarRef
    value: Expr


@dataclass(slots=True)
class Attribute(Expr):
    ''' Attribute access expression. '''
    class_name: str
    target: VarRef
    attr: str


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
    then_block: _List[Expr]
    else_block: _List[Expr] | None


@dataclass(slots=True)
class While(Expr):
    cond: VarDecl | Expr
    body: Let


@dataclass(slots=True)
class Struct(Node):
    ''' Struct is a cl-struct. '''
    name: str
    fields: _List[VarDecl]

    # These are global functions.
    methods: _List[Function] = field(default_factory=list)


@dataclass(slots=True)
class Block(Node):
    ''' Block is a list of expressions. '''
    name: str
    stmts: _List[Expr]
