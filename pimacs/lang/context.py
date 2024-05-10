from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import *

import pimacs.lang.ir as ir
import pimacs.lang.type as _ty


class ModuleContext:
    """Context is a class that represents the context of the Module.
    The states includs
        - function symbols
        - variable symbols
        - type symbols
        - class symbols

    The module context could be nested.
    """

    def __init__(self, name: str):
        self._name = name
        self._functions: Dict[str, ir.FuncDecl] = {}
        self._variables: Dict[str, ir.VarDecl] = {}
        self._classes: Dict[str, ir.ClassDef] = {}
        self._types: Dict[str, _ty.Type] = {}

    def get_symbol(
        self, name: str
    ) -> Optional[Union[ir.FuncDecl, ir.VarDecl, ir.ClassDef]]:
        if name in self._functions:
            return self._functions[name]
        if name in self._variables:
            return self._variables[name]
        if name in self._classes:
            return self._classes[name]
        return None

    def get_type(
        self, name: str, subtypes: Optional[List[Type]] = None
    ) -> Optional[_ty.Type]:
        key = f"{name}[{', '.join(map(str, subtypes))}]" if subtypes else name
        if key in self._types:
            return self._types[key]
        new_type = _ty.make_customed(name, subtypes)
        self._types[key] = new_type
        return new_type

    def symbol_exists(self, name: str) -> bool:
        return (
            name in self._functions or name in self._variables or name in self._classes
        )

    def add_function(self, func: ir.FuncDecl):
        self._functions[func.name] = func

    def add_variable(self, var: ir.VarDecl):
        self._variables[var.name] = var

    def add_class(self, cls: ir.ClassDef):
        self._classes[cls.name] = cls

    def get_function(self, name: str) -> Optional[ir.FuncDecl]:
        return self._functions.get(name)

    def get_variable(self, name: str) -> Optional[ir.VarDecl]:
        return self._variables.get(name)

    def get_class(self, name: str) -> Optional[ir.ClassDef]:
        return self._classes.get(name)

    @property
    def name(self) -> str:
        """Module name."""
        return self._name


@dataclass(unsafe_hash=True)
class Symbol:
    """
    Reprsent any kind of symbol and is comparable.
    """

    class Kind(Enum):
        Unk = -1
        Func = 0
        Class = 1
        Member = 2  # class member
        Var = 3  # normal variable
        Lisp = 4
        Arg = 5

        def __str__(self):
            return self.name

    name: str  # the name without "self." prefix if it is a member
    kind: Kind

    def __str__(self):
        return f"{self.kind.name}({self.name})"


SymbolItem = Any


@dataclass
class Scope:
    data: Dict[Symbol, SymbolItem] = field(default_factory=dict)

    class Kind(Enum):
        Local = 0
        Global = 1
        Class = 2
        Func = 3

    kind: Kind = Kind.Local

    def add(self, symbol: Symbol, item: SymbolItem):
        if symbol in self.data:
            raise KeyError(f"{item.loc}\nSymbol {symbol} already exists")
        self.data[symbol] = item

    def get(self, symbol: Symbol) -> SymbolItem | None:
        return self.data.get(symbol, None)


class SymbolTable:
    def __init__(self):
        self.scopes = [Scope(kind=Scope.Kind.Global)]

    def push_scope(self, kind: Scope.Kind):
        self.scopes.append(Scope(kind=kind))

    def pop_scope(self):
        self.scopes.pop()

    def add_symbol(self, symbol: Symbol, item: SymbolItem):
        self.scopes[-1].add(symbol=symbol, item=item)
        return item

    def get_symbol(
        self,
        symbol: Optional[Symbol] = None,
        name: Optional[str] = None,
        kind: Optional[Symbol.Kind | List[Symbol.Kind]] = None,
    ) -> Optional[SymbolItem]:
        symbols = {symbol}
        if not symbol:
            assert name and kind
            symbols = (
                {Symbol(name=name, kind=kind)}
                if isinstance(kind, Symbol.Kind)
                else {Symbol(name=name, kind=k) for k in kind}
            )

        for symbol in symbols:
            for scope in reversed(self.scopes):
                ret = scope.get(symbol)
                if ret:
                    return ret
        return None

    def contains(self, symbol: Symbol) -> bool:
        return any(symbol in scope for scope in reversed(self.scopes))

    def contains_locally(self, symbol: Symbol) -> bool:
        return self.scopes[-1].get(symbol) is not None

    @property
    def current_scope(self):
        return self.scopes[-1]

    @contextmanager
    def scope_guard(self, kind=Scope.Kind.Local):
        self.push_scope(kind)
        try:
            yield
        finally:
            self.pop_scope()
