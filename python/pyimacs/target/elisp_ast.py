import abc
import sys
from dataclasses import dataclass
from typing import *

# This file defines some speicfic syntax AST for Elisp.


class Dumper:
    def __init__(self, io: Any) -> None:
        self.io = io
        self.indent = 0
        self.indent_unit = 4

    def do_indent(self, indent: int) -> None:
        self.indent += self.indent_unit

    def undo_indent(self, indent: int) -> None:
        self.indent -= self.indent_unit

    def print(self, s: str) -> None:
        self.io.write(" " * self.indent + s)

    def println(self, s: str) -> None:
        self.print(s)
        self.print('\n')


class Node:
    @abc.abstractclassmethod
    def dump(self, dumper: Dumper) -> None:
        pass

    def __str__(self) -> str:
        dumper = Dumper(StrIO())
        return self.dump(dumper)


@dataclass()
class Token(Node):
    '''
    Token helps to represent constants(int, float, string, symbol) or variables in elisp.
    '''
    symbol: Any
    is_symbol: bool = False

    def is_int(self) -> bool:
        return type(self.symbol) is int

    def is_float(self) -> bool:
        return type(self.symbol) is float

    def is_string(self) -> bool:
        return type(self.symbol) is str and (not self.is_symbol)

    def dump(self, dumper: Dumper) -> None:
        if self.is_symbol:
            assert type(self.symbol) is str
            dumper.print(self.symbol)
        elif type(self.symbol) is str:
            dumper.print(f'"{self.symbol}"')
        else:
            dumper.print(self.symbol)


class Symbol(Token):
    def __init__(self, symbol: str):
        super().__init__(symbol, is_symbol=True)


@dataclass()
class Var(Node):
    name: str
    default: Any = None

    def dump(self, dumper: Dumper) -> None:
        dumper.print(self.name)


class Expr(abc.ABC):
    ''' A (...) expression '''
    @property
    @abc.abstractclassmethod
    def symbols(self) -> List[Any]:
        raise NotImplementedError()

    def dump(self, dumper: Dumper) -> None:
        dumper.print(f"( {' '.join([str(s) for s in self.symbols])} )")


class Expression(Expr):
    def __init__(self, *symbols: List[Any]):
        self._symbols = symbols

    @property
    def symbols(self):
        return self._symbols


@dataclass()
class LetExpr(Expr):
    vars: List[Var]
    body: List[Expr]

    @property
    def symbols(self) -> List[Any]:
        return [Symbol("let"), Expression(*self.vars), Expression(*self.body)]

    def dump(self, dumper: Dumper) -> None:
        dumper.println("(let* ")
        dumper.do_indent()
        def repr_var(
            var): return f"({var.name} {var.default})" if var.default else var.name
        dumper.println(f"({' '.join([repr_var(v) for v in self.vars])})")
        for b in self.body:
            b.dump(dumper)
            dumper.println
        dumper.undo_indent()
        dumper.println(")")


@dataclass()
class Function(Node):
    name: str
    args: List[Var]
    body: LetExpr

    def dump(self, dumper: Dumper) -> None:
        dumper.println(
            f"(defun {self.name} ({' '.join([str(a) for a in self.args])})")
        dumper.do_indent()
        self.body.dump(dumper)
        dumper.undo_indent()
        dumper.println(")")


class Call(Expr):
    def __init__(self, name: str, args: List[Var]):
        self.symbols = [name, *args]

    def dump(self, dumper: Dumper) -> None:
        dumper.print("(")
        for s in self.symbols[:-1]:
            s.dump(dumper)
            dumper.print(" ")
        self.symbols[-1].dump(dumper)
        dumper.print(")")


class IfElse(Node):
    cond: Var
    then_body: LetExpr
    else_body: LetExpr

    def dump(self, dumper: Dumper) -> None:
        dumper.println("(if")
        dumper.do_indent()
        self.cond.dump(dumper)
        dumper.println()
        self.then_body.dump(dumper)
        dumper.println()
        self.else_body.dump(dumper)
        dumper.undo_indent()
        dumper.println(")")


class While:
    cond: Var
    body: LetExpr

    def dump(self, dumper: Dumper) -> None:
        dumper.println("(while")
        dumper.do_indent()
        self.cond.dump(dumper)
        dumper.println()
        self.body.dump(dumper)
        dumper.undo_indent()
        dumper.println(")")


class StrIO:
    def __init__(self) -> None:
        self._content = ""

    def write(self, c: str) -> None:
        self._content += c

    def content(self) -> str:
        return self._content
