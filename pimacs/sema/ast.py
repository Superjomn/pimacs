'''
This file contains several additional AST node for semantic analysis.
'''
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import pimacs.ast.ast as ast

from .func import FuncOverloads, FuncSymbol
from .symbol_table import Scope
from .utils import Symbol


@dataclass(slots=True, unsafe_hash=True)
class AnalyzedClass(ast.Class):
    # symbols hold all the members and methods of the class
    # It should be updated once the class is modified
    symbols: Scope = field(default_factory=Scope, init=False, hash=False)

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
        self.symbols = Scope()
        for node in self.body:
            if isinstance(node, ast.VarDecl):
                symbol = Symbol(name=node.name, kind=Symbol.Kind.Member)
                self.symbols.add(symbol, node)
            elif isinstance(node, ast.Function):
                self.symbols.add(FuncSymbol(node.name), node)


@dataclass(slots=True)
class MakeObject(ast.Expr):
    '''
    Create an object from a class.

    e.g.
      App()      # => MakeObject(class_name='App')
    '''
    class_name: str

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass

    def __str__(self):
        return f"{self.class_name}()"


@dataclass
class UCallAttr(ast.Call):
    '''
    Call an attribute of an object.

    It will be replaced to a actual method call in the Sema.

    e.g.
      app = App()
      app.time()      # => UCallAttr(app, attr='time')
    '''
    obj: Optional[ast.Expr] = None
    attr: str = ""

    resolved: bool = field(default=False, init=False)


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
