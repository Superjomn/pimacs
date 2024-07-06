'''
This file contains several additional AST nodes dedicated to semantic analysis.
'''
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import pimacs.ast.type as ty
from pimacs.ast.ast import *

from .func import FuncSymbol
from .symbol_table import Scope
from .utils import Symbol


@dataclass(slots=True, unsafe_hash=True)
class AnalyzedClass(Class):
    # symbols hold all the members and methods of the class
    # It should be updated once the class is modified
    symbols: Scope = field(default_factory=Scope,
                           init=False, hash=False, repr=False)

    @classmethod
    def create(cls, node: Class) -> "AnalyzedClass":
        return cls(
            name=node.name,
            body=node.body,
            decorators=node.decorators,
            template_params=node.template_params,
            loc=node.loc,
        )

    @staticmethod
    def _extract_template_params(node: Class):
        if not node.decorators:
            return tuple()
        for dec in node.decorators:
            if isinstance(dec.action, Template):
                return dec.action.types
        return tuple()

    @contextmanager
    def auto_update_symbols(self):
        yield
        self.update_symbols()

    def update_symbols(self):
        self.symbols = Scope()
        for node in self.body:
            if isinstance(node, VarDecl):
                symbol = Symbol(name=node.name, kind=Symbol.Kind.Member)
                self.symbols.add(symbol, node)
            elif isinstance(node, Function):
                self.symbols.add(FuncSymbol(node.name), node)


@dataclass(slots=True, unsafe_hash=True)
class MakeObject(Expr):
    '''
    Create an object from a class.

    e.g.
      App()      # => MakeObject(class_name='App')
    '''

    def _refresh_users(self):
        pass

    def replace_child(self, old, new):
        pass

    def __str__(self):
        return f"{self.type.name}()"


@dataclass
class UCallMethod(Unresolved, Expr):
    '''
    Call a method of an object.

    It will be replaced to a actual method call in the Sema.

    e.g.
      app = App()
      app.time()      # => UCallAttr(app, attr='time')
    '''
    obj: Optional[Union[CallParam, "UModule"]] = None  # self
    attr: str = ""
    args: Tuple[CallParam, ...] = field(default_factory=tuple)

    type_spec: Tuple[ty.Type, ...] = field(default_factory=tuple)

    def __post_init__(self):
        self._refresh_users()

    def _refresh_users(self):
        if self.obj:
            print(f"refreshing users of {self}, obj: {
                  self.obj}, type: {type(self.obj)}")
            self.obj.add_user(self)
        for arg in self.args:
            arg.add_user(self)

    def replace_child(self, old, new):
        if self.obj == old:
            self.obj = new
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new


@dataclass(slots=True)
class CallMethod(Expr):
    '''
    Call an method of an object.

    e.g.
      app = App()
      app.time()      # => CallAttr(app, attr='time')
    '''
    obj: CallParam | UModule  # The object to call
    method: Function
    args: Tuple[CallParam, ...]

    def __post_init__(self):
        self._refresh_users()

    def _refresh_users(self):
        self.users.clear()
        self.obj.add_user(self)
        for arg in self.args:
            arg.add_user(self)

    def replace_child(self, old, new):
        if self.obj == old:
            self.obj = new
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new
