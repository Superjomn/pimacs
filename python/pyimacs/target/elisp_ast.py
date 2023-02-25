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
        # True since the first line should not indent by default.
        self._indent_processed = True

    def do_indent(self, indent: int = 1) -> None:
        self.indent += indent

    def undo_indent(self, indent: int = 1) -> None:
        self.indent -= indent

    def put(self, s: str) -> None:
        ''' Simply put the a string to the IO. '''
        self.io.write(s)

    def print(self, s: str) -> None:
        ''' Smart print with indent considered.
        Note, if a line break is needed, use println instead. Never use something like print('\n').
        '''
        if not self._indent_processed:
            self.put(self.indent * self.indent_unit * " ")
            self._indent_processed = True
        self.put(s)

    def println(self, s: str = "") -> None:
        ''' Print with a tailing line break. '''
        if s:
            self.print(s + "\n")
        else:
            self.put("\n")
        self._indent_processed = False


class Node:
    @abc.abstractclassmethod
    def dump(self, dumper: Dumper) -> None:
        pass

    def __str__(self) -> str:
        io = StrIO()
        dumper = Dumper(io)
        self.dump(dumper)
        return io.content


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
            dumper.put(self.symbol)
        elif type(self.symbol) is str:
            dumper.put(f'"{self.symbol}"')
        else:
            dumper.put(str(self.symbol))


class Symbol(Token):
    def __init__(self, symbol: str):
        super().__init__(symbol, is_symbol=True)


@dataclass()
class Var(Node):
    name: str
    default: Any = None

    def dump(self, dumper: Dumper) -> None:
        dumper.put(self.name)


class Expr(abc.ABC, Node):
    ''' Base class of all the expressios. '''
    @property
    @abc.abstractclassmethod
    def symbols(self) -> List[Any]:
        raise NotImplementedError()

    def dump(self, dumper: Dumper) -> None:
        dumper.print("(")
        n = len(self.symbols)
        for id, sym in enumerate(self.symbols):
            sym.dump(dumper)
            if id != n-1:
                dumper.put(" ")
        dumper.print(")")


class Expression(Expr):
    ''' An expression, it will dumped like (a b c) '''

    def __init__(self, *symbols: List[Any]):
        for x in symbols:
            assert isinstance(x, Node), f"{x} is not a AST node"

        self._symbols = symbols

    @property
    def symbols(self):
        return self._symbols


@dataclass()
class LetExpr(Expr):
    ''' Let expression. '''
    vars: List[Var]
    body: List[Expr]

    @property
    def symbols(self) -> List[Any]:
        return [Symbol("let"), Expression(*self.vars), Expression(*self.body)]

    def dump(self, dumper: Dumper) -> None:
        dumper.println("(let*")
        dumper.do_indent()

        def repr_var(var):
            return f"({str(var)} {var.default})" if var.default is not None else str(var)
        dumper.println(f"({' '.join([repr_var(v) for v in self.vars])})")
        for b in self.body:
            dumper.print("")
            b.dump(dumper)
            dumper.println()
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


@dataclass
class Call(Expr):
    func: Symbol
    args: List[Var]

    @property
    def symbols(self):
        return [self.func, *self.args]


@dataclass
class IfElse(Node):
    cond: Var
    then_body: LetExpr
    else_body: LetExpr

    def dump(self, dumper: Dumper) -> None:
        dumper.print(f"(if ")
        self.cond.dump(dumper)
        dumper.println()
        dumper.do_indent()
        self.then_body.dump(dumper)
        dumper.println()
        self.else_body.dump(dumper)
        dumper.undo_indent()
        dumper.print(")")


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

    @property
    def content(self) -> str:
        return self._content
