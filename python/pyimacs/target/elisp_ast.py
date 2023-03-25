import abc
import logging
import sys
from dataclasses import dataclass, field
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
    # the nodes used this
    def __init__(self) -> None:
        self.users = set()

    @abc.abstractclassmethod
    def dump(self, dumper: Dumper) -> None:
        pass

    def __str__(self) -> str:
        io = StrIO()
        dumper = Dumper(io)
        self.dump(dumper)
        return io.content

    def add_user(self, user: "Node") -> None:
        logging.debug(f"add_user {self} -> {user}")
        self.users.add(user)

    def del_user(self, user: "Node") -> None:
        self.users.remove(user)

    @property
    def user_count(self) -> int:
        return len(self.users)

    def replace_symbol(self, old: Any, new: Any) -> None:
        raise NotImplementedError()


@dataclass
class Token(Node):
    '''
    Token helps to represent constants(int, float, string, symbol) or variables in elisp.
    '''
    symbol: Any
    is_symbol: bool = False

    def __init__(self, symbol: Any, is_symbol: bool = False):
        super().__init__()
        while isinstance(symbol, Token):
            symbol = symbol.symbol
        self.symbol = symbol
        self.is_symbol = is_symbol
        assert isinstance(symbol, (str, int, float)
                          ), f"Invalid token {symbol} of type {type(symbol)}"

    def is_int(self) -> bool:
        return type(self.symbol) is int

    def is_float(self) -> bool:
        return type(self.symbol) is float

    def is_string(self) -> bool:
        return type(self.symbol) is str and (not self.is_symbol)

    def dump(self, dumper: Dumper) -> None:
        if self.is_symbol:
            dumper.put(self.symbol)
        elif type(self.symbol) is str:
            dumper.put(f'"{self.symbol}"')
        else:
            dumper.put(str(self.symbol))

    def replace_symbol(self, old: Any, new: Any) -> None:
        pass


@dataclass
class Symbol(Token):
    def __init__(self, symbol: str):
        super().__init__(symbol, is_symbol=True)


@dataclass
class Var(Node):
    name: str
    default: Any = None

    def __init__(self, name: str, default: Any = None):
        super().__init__()
        self.name = name
        self.default = default

    def dump(self, dumper: Dumper) -> None:
        dumper.put(self.name)

    def replace_symbol(self, old: Any, new: Any) -> None:
        pass

    def __hash__(self) -> int:
        return hash(self.name)


class Expr(abc.ABC, Node):
    ''' Base class of all the expressios. '''

    def __init__(self) -> None:
        super().__init__()

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
        super().__init__()
        for x in symbols:
            assert isinstance(x, Node), f"{x} is not a AST node"

        self._symbols = [*symbols]

        syms = self._symbols
        if self.symbols[0] == Symbol("setq"):
            syms = syms[2:]
        for sym in syms:
            sym.add_user(self)

    def append(self, sym: Any) -> None:
        self._symbols.append(sym)
        sym.add_user(self)

    @property
    def symbols(self):
        return self._symbols

    def replace_symbol(self, old: Any, new: Any) -> None:
        assert old is not None
        assert new is not None
        symbols = [x for x in self._symbols]
        for i, sym in enumerate(symbols):
            if sym == old:
                symbols[i] = new
                if self in old.users:
                    old.del_user(self)
                    new.add_user(self)
            else:
                sym.replace_symbol(old, new)
        self._symbols = symbols


@dataclass()
class LetExpr(Expr):
    ''' Let expression. '''
    vars: List[Var]
    body: List[Expr]

    def __init__(self, vars: List[Var], body: List[Expr]):
        super().__init__()
        self.vars = vars
        self.body = body

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

    def replace_symbol(self, old: Any, new: Any) -> None:
        vars = [x for x in filter(lambda x: x != old, self.vars)]
        if len(vars) == len(self.vars):  # nothing changed
            return
        for expr in self.body:
            expr.replace_symbol(old, new)

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class Guard(Expr):
    name: str
    args: List[Var]
    body: List[Expr]

    def __init__(self, name: str, args: List[Var], body: List[Expr]):
        super().__init__()
        self.name = name
        self.args = args
        self.body = body

    @property
    def symbols(self) -> List[Any]:
        return [Symbol(self.name), *self.body]

    def dump(self, dumper: Dumper) -> None:
        dumper.println(f"({self.name}")
        dumper.do_indent()
        body = self.body
        if not isinstance(body, Iterable):
            body = [body]
        for b in body:
            b.dump(dumper)
        dumper.undo_indent()
        dumper.print(")")

    def replace_symbol(self, old: Any, new: Any) -> None:
        for expr in self.body:
            expr.replace_symbol(old, new)

    def __hash__(self) -> int:
        return hash(str(self))


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

    def replace_symbol(self, old: Any, new: Any) -> None:
        for expr in self.body:
            expr.replace_symbol(old, new)


@dataclass
class Call(Expr):
    func: Symbol
    args: List[Var]
    is_void_call: bool

    def __init__(self, func: Symbol, args: List[Var], is_void_call: bool = False):
        super().__init__()
        self.func = func
        self.args = args

        self.func.add_user(self)
        for arg in self.args:
            arg.add_user(self)
        self.is_void_call = is_void_call

    @property
    def symbols(self):
        return [self.func, *self.args]

    def replace_symbol(self, old: Any, new: Any) -> None:
        for id, arg in enumerate(self.args):
            if arg == old:
                self.args[id] = new
                if self in old.users:
                    old.del_user(self)
                    new.add_user(self)
            else:
                arg.replace_symbol(old, new)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class IfElse(Node):
    cond: Var
    then_body: LetExpr
    else_body: LetExpr

    def __init__(self, cond: Var, then_body: LetExpr, else_body: LetExpr = None):
        super().__init__()

        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

        self.cond.add_user(self)
        self.then_body.add_user(self)
        if self.else_body:
            self.else_body.add_user(self)

    def dump(self, dumper: Dumper) -> None:
        dumper.print(f"(if ")
        self.cond.dump(dumper)
        dumper.println()
        dumper.do_indent()
        self.then_body.dump(dumper)
        dumper.println()
        if self.else_body:
            self.else_body.dump(dumper)
        dumper.undo_indent()
        dumper.print(")")

    def replace_symbol(self, old: Any, new: Any) -> None:
        self.cond.replace_symbol(old, new)
        self.then_body.replace_symbol(old, new)
        if self.else_body:
            self.else_body.replace_symbol(old, new)

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class While(Node):
    cond: Var
    body: LetExpr

    def __init__(self, cond: Var, body: LetExpr):
        super().__init__()

        self.cond = cond
        self.body = body

        self.cond.add_user(self)
        self.body.add_user(self)

    def dump(self, dumper: Dumper) -> None:
        dumper.println("(while")
        dumper.do_indent()
        self.cond.dump(dumper)
        dumper.println()
        self.body.dump(dumper)
        dumper.undo_indent()
        dumper.println(")")

    def replace_symbol(self, old: Any, new: Any) -> None:
        self.cond.replace_symbol(old, new)
        self.body.replace_symbol(old, new)


class StrIO:
    def __init__(self) -> None:
        self._content = ""

    def write(self, c: str) -> None:
        self._content += c

    @property
    def content(self) -> str:
        return self._content
