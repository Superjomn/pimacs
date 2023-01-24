from dataclasses import dataclass
from typing import *

# This file defines some speicfic syntax AST for Elisp.


@dataclass()
class Token:
    symbol: Any
    is_symbol: bool = False

    def is_int(self) -> bool:
        return type(self.symbol) is int

    def is_float(self) -> bool:
        return type(self.symbol) is float

    def is_string(self) -> bool:
        return type(self.symbol) is str and (not self.is_symbol)


@dataclass()
class Var:
    name: str
    default: Any = None


@dataclass()
class Expr:
    ''' A (...) expression '''
    symbols: List[Any]


@dataclass()
class LetExpr:
    vars: List[Var]
    body: List[Expr]


@dataclass()
class Function:
    name: str
    args: List[Var]
    body: LetExpr


class Call(Expr):
    def __init__(self, name: str, args: List[Var]):
        self.symbols = [name, *args]


class IfElse:
    cond: Var
    then_body: LetExpr
    else_body: LetExpr


class While:
    cond: Var
    body: LetExpr
