from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from tabulate import tabulate  # type: ignore

from pimacs.ast import ast

from .func import FuncOverloads
from .utils import (ClassId, FuncSymbol, ModuleId, Scoped, ScopeKind, Symbol,
                    SymbolItem, bcolors, print_colored)


class Scope:
    def __init__(self, kind: ScopeKind = ScopeKind.Local, parent: Optional["Scope"] = None):
        self.data: Dict[Symbol, SymbolItem] = {}
        self.kind = kind
        self.parent = parent

    def add(self, symbol: Symbol, item: SymbolItem):
        if symbol.kind == Symbol.Kind.Func:
            self._add_func(item)
        else:
            self._add_symbol(symbol, item)

    def get(self, symbol: Symbol) -> SymbolItem | None:
        if symbol.kind == Symbol.Kind.Func:
            return self._get_func(symbol)
        return self._get_symbol(symbol)

    def get_local(self, symbol: Symbol) -> SymbolItem | None:
        return self.data.get(symbol, None)

    def _add_symbol(self, symbol: Symbol, item: SymbolItem):
        ''' Add non-func record. '''
        if symbol in self.data:
            raise KeyError(f"Symbol {symbol} already exists")
        self.data[symbol] = item

    def _get_symbol(self, symbol: Symbol) -> SymbolItem | None:
        ''' Get non-func record. '''
        x: Scope = self
        while x:
            if tmp := x.get_local(symbol):
                return tmp
            x = x.parent
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

        # TODO: optimize the performance
        if overloads:
            tmp = overloads.pop()
            for overload in overloads:
                tmp += overload
            return tmp
        return None

    def _add_func(self, func: ast.Function):
        ''' Add function record. '''
        symbol = FuncSymbol(func.name)
        record = self._get_func(symbol)
        if record:
            record.insert(func)
        else:
            record = FuncOverloads(symbol)
            record.insert(func)
            self.add(symbol, record)

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
        self.scopes = [Scope(kind=ScopeKind.Global)]

    def push_scope(self, kind: ScopeKind = ScopeKind.Local):
        new_scope = Scope(kind=kind, parent=self.scopes[-1])
        self.scopes.append(new_scope)

    def pop_scope(self):
        self.scopes.pop()

    def insert(self, symbol: Symbol, item: SymbolItem):
        self.scopes[-1].add(symbol, item)

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
            if ret := self.scopes[-1].get(symbol):
                return ret
        return None

    def get_function(self, symbol: FuncSymbol) -> Optional[FuncOverloads]:
        return self.scopes[-1].get(symbol)

    def contains(self, symbol: Symbol) -> bool:
        return self.get_symbol(symbol) is not None

    def contains_locally(self, symbol: Symbol) -> bool:
        return self.scopes[-1].get_local(symbol) is not None

    @property
    def current_scope(self):
        return self.scopes[-1]

    @contextmanager
    def scope_guard(self, kind=ScopeKind.Local):
        self.push_scope(kind)
        try:
            yield
        finally:
            self.pop_scope()

    def print_summary(self):
        table = [["Symbol", "Kind", "Summary"]]
        for no, scope in enumerate(self.scopes):
            for symbol, item in scope.data.items():
                table.append([symbol.name, symbol.kind, str(item)])
            for symbol, item in self.func_table.scopes[no].data.items():
                table.append([symbol.name, symbol.kind, str(item)])

            print_colored(f"Scope {no}:\n", bcolors.OKBLUE)
            print_colored(
                tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
            print_colored("\n")
