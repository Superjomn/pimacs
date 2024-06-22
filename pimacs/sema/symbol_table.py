from typing import Dict, List, Optional

# from multidispatch import dispatch
from multimethod import multimethod
from tabulate import tabulate  # type: ignore

from pimacs.ast import ast

from .func import FuncOverloads
from .utils import (FuncSymbol, Scoped, ScopeKind, Symbol, SymbolItem, bcolors,
                    print_colored)


class Scope:
    def __init__(self, kind: ScopeKind = ScopeKind.Local, parent: Optional["Scope"] = None):
        self.data: Dict[Symbol, SymbolItem] = {}
        self.kind = kind
        self.parent = parent

    def add(self, symbol: Symbol, item: SymbolItem):
        if symbol.kind == Symbol.Kind.Func:
            self._add_func(symbol, item)
        else:
            self._add_symbol(symbol, item)

    @multimethod
    def get(self, symbol: Symbol) -> SymbolItem | None:
        return self._get_symbol(symbol)

    @multimethod  # type: ignore
    def get(self, kind: Symbol.Kind) -> List[SymbolItem]:
        return [item for symbol, item in self.data.items() if symbol.kind == kind]

    def get_local(self, symbol: Symbol) -> SymbolItem | None:
        return self.data.get(symbol, None)

    def _add_symbol(self, symbol: Symbol, item: SymbolItem):
        ''' Add non-func record. '''
        if symbol in self.data:
            raise KeyError(f"Symbol {symbol} already exists")
        self.data[symbol] = item

    def _get_symbol(self, symbol: Symbol) -> SymbolItem | None:
        ''' Get non-func record. '''
        scope: Scope = self
        while scope is not None:
            if tmp := scope.get_local(symbol):
                return tmp
            scope = scope.parent  # type: ignore
        return None

    def _get_func(self, symbol: Symbol) -> Optional[FuncOverloads]:
        ''' Get function record. '''
        # get a FuncOverloads holding all the functions with the same symbol
        overloads: List[FuncOverloads] = []
        scope = self
        while scope:
            record = scope.get_local(symbol)
            if record:
                overloads.append(record)
            scope = scope.parent  # type: ignore

        # TODO: optimize the performance
        if overloads:
            tmp = overloads.pop()
            for overload in overloads:
                tmp += overload
            return tmp
        return None

    def _add_func(self, symbol: Symbol, func: ast.Function):
        ''' Add function record. '''
        record = self._get_func(symbol)
        if record:
            record.insert(func)
        else:
            record = FuncOverloads(symbol)
            record.insert(func)
            self.data[symbol] = record

    def __contains__(self, symbol: Symbol) -> bool:
        return self.get(symbol) is not None

    def __len__(self) -> int:
        return len(self.data)

    def print_summary(self):
        table = [["Symbol", "Kind", "Summary"]]
        for symbol, item in self.data.items():
            table.append([symbol.name, symbol.kind, str(item)[:50]])

        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


class SymbolTable(Scoped):
    def __init__(self):
        self._scopes = [Scope(kind=ScopeKind.Global)]

    @property
    def global_scope(self) -> Scope:
        return self._scopes[0]

    @property
    def current_scope(self) -> Scope:
        return self._scopes[-1]

    @multimethod
    def insert(self, symbol: Symbol, item: SymbolItem):
        self.current_scope.add(symbol, item)
        return item

    @multimethod  # type: ignore
    def insert(self, func: ast.Function):
        symbol = FuncSymbol(func.name)
        self.current_scope.add(symbol, func)
        return func

    @multimethod
    def lookup(self, symbol: Symbol) -> Optional[SymbolItem]:
        return self.current_scope.get(symbol)

    @multimethod  # type: ignore
    def lookup(self, symbols: List[Symbol]) -> Optional[SymbolItem]:
        for symbol in symbols:
            if ret := self.current_scope.get(symbol):
                return ret

    @multimethod  # type: ignore
    def lookup(self, name: str, kind: Symbol.Kind) -> Optional[SymbolItem]:
        return self.lookup(Symbol(name=name, kind=kind))

    @multimethod  # type: ignore
    def lookup(self, name: str, kind: List[Symbol.Kind]) -> Optional[SymbolItem]:
        for k in kind:
            if ret := self.lookup(name, k):
                return ret

    def contains(self, symbol: Symbol) -> bool:
        return self.lookup(symbol) is not None

    def contains_locally(self, symbol: Symbol) -> bool:
        return self.current_scope.get_local(symbol) is not None

    def print_summary(self):
        table = [["Symbol", "Kind", "Summary"]]
        for no, scope in enumerate(self._scopes):
            print_colored(f"Scope {no} {scope.kind.name}:\n", bcolors.OKBLUE)
            scope.print_summary()

    def push_scope(self, kind: ScopeKind = ScopeKind.Local):
        new_scope = Scope(kind=kind, parent=self.current_scope)
        assert len(new_scope) == 0
        self._scopes.append(new_scope)

        return self.current_scope

    def pop_scope(self):
        self._scopes.pop()
