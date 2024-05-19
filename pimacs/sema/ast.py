'''
This file contains several additional AST node for semantic analysis.
'''
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

import pimacs.ast.ast as ast

from .func import FuncOverloads, FuncSymbol
from .utils import Symbol


@dataclass(slots=True, unsafe_hash=True)
class AnalyzedClass(ast.Class):
    # symbols hold all the members and methods of the class
    # It should be updated once the class is modified
    symbols: Dict[Symbol, ast.VarDecl | FuncOverloads] = field(
        default_factory=dict, init=False, hash=False)

    @classmethod
    def create(cls, node: ast.Class) -> "AnalyzedClass":
        return cls(
            name=node.name,
            body=node.body,
            loc=node.loc,
        )

    @contextmanager
    def auto_update_symbols(self):
        yield
        self.update_symbols()

    def update_symbols(self):
        self.symbols = {}
        for node in self.body:
            if isinstance(node, ast.VarDecl):
                symbol = Symbol(name=node.name, kind=Symbol.Kind.Member)
                self.symbols[symbol] = node
            elif isinstance(node, ast.Function):
                symbol = FuncSymbol(node.name)
                self.symbols.get(symbol, FuncOverloads(
                    symbol=symbol)).insert(node)


@dataclass(slots=True)
class GetAttr(ast.Node):
    '''
    Get an attribute from an object.

    e.g.
      app = App()
      app.time      # => GetAttr(obj=app, attr='time')
    '''
    obj: ast.Expr
    attr: str

    def __str__(self):
        return f"{self.obj}.{self.attr}"
