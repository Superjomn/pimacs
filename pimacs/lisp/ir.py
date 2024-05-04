from dataclasses import dataclass, field
from typing import List as Li
from typing import Optional, Union

import pimacs.lang.ir as ir


@dataclass(slots=True)
class IrNode:
    loc: Optional[ir.Location] = field(compare=False, repr=False)

@dataclass(slots=True)
class Expr(IrNode):
    pass

# Basic data structures
@dataclass(slots=True)
class Symbol(Expr):
    name: str

@dataclass(slots=True)
class Var(Expr):
    '''
    Variable
    '''
    name: str

@dataclass(slots=True)
class List(Expr):
    '''
    List expression
    '''
    elements: Li[Expr]

class PList(Expr):
    '''
    Parentheses list
    '''
    elements: Li[Expr]

class HashTable(Expr):
    '''
    Hash table
    '''
    elements: Li[ir.Expr]

class Vector(ir.Expr):
    '''
    Vector
    '''
    elements: Li[ir.Expr]

# IR Nodes

@dataclass(slots=True)
class Module(IrNode):
    name: str
    body: Li[Expr] = field(default_factory=list)

@dataclass(slots=True)
class Argument(IrNode):
    '''
    Argument for let, func
    '''
    name: str

class Block(Expr):
    '''
    Block of statements
    '''
    docs : str = ""
    stmts: Li[Expr]

    def append(self, stmt:Expr):
        self.stmts.append(stmt)

@dataclass(slots=True)
class SetqStmt(Expr):
    '''
    Setq statement
    '''
    target: Var
    value: Expr

class SetStmt(Expr):
    '''
    Set statement
    '''
    target: ir.VarRef
    value: ir.Expr

@dataclass(slots=True)
class Func(IrNode):
    '''
    For defun.
    '''
    name: str
    body: Block

class FuncCall(ir.Stmt):
    '''
    Function call
    '''
    func: str
    args: Li[ir.Expr]

@dataclass
class LetStmt(ir.Stmt):
    args: Li[Argument]
    body: ir.Block
