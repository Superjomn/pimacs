from typing import Dict, List, Optional, Tuple

# from multidispatch import dispatch
from multimethod import multimethod
from tabulate import tabulate  # type: ignore

from pimacs.ast import ast

from .func import FuncOverloads
from .utils import (FuncSymbol, Scoped, ScopeKind, Symbol, SymbolItem, bcolors,
                    print_colored)


class Scope:
    def __init__(self, kind: ScopeKind = ScopeKind.Local, parent: Optional["Scope"] = None):
        # format: {symbol: (item, distance)}, the distance is the level of scopes to the current scope
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
        ret = self.get_symbol_with_distance(symbol)
        if ret:
            return ret[0]
        return None

    @multimethod  # type: ignore
    def get(self, kind: Symbol.Kind) -> List[SymbolItem]:
        return [item for symbol, item in self.data.items() if symbol.kind == kind]

    @multimethod  # type: ignore
    def get_local(self, symbol: Symbol) -> SymbolItem | None:
        return self.data.get(symbol, None)

    @multimethod  # type: ignore
    def get_local(self, kind: Symbol.Kind) -> List[SymbolItem]:
        if kind == Symbol.Kind.Func:
            func_overloads = [item for symbol,
                              item in self.data.items() if symbol.kind == kind]
            funcs = []
            for overloads in func_overloads:
                funcs.extend(overloads.funcs.values())
            return funcs
        else:
            return [item for symbol, item in self.data.items() if symbol.kind == kind]

    def update_local(self, symbol: Symbol, item: SymbolItem):
        ''' Update the local symbol table with the new item. '''
        assert self.get_local(symbol) is not None
        self.data[symbol] = item

    def _add_symbol(self, symbol: Symbol, item: SymbolItem):
        ''' Add non-func record. '''
        if symbol in self.data:
            raise KeyError(f"Symbol {symbol} already exists")
        self.data[symbol] = item

    def get_symbol_with_distance(self, symbol: Symbol) -> Tuple[SymbolItem, int] | None:
        ''' Get non-func record. '''
        scope: Scope = self
        level = 0
        while scope is not None:
            if tmp := scope.get_local(symbol):
                return tmp, level
            level += 1
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
        items = []
        for symbol in symbols:
            if item := self.current_scope.get_symbol_with_distance(symbol):
                items.append(item)
        items = sorted(items, key=lambda x: x[1])
        return items[0][0] if items else None

    @multimethod  # type: ignore
    def lookup(self, name: str, kind: Symbol.Kind) -> Optional[SymbolItem]:
        return self.lookup(Symbol(name=name, kind=kind))

    @multimethod  # type: ignore
    def lookup(self, name: str, kind: List[Symbol.Kind]) -> Optional[SymbolItem]:
        symbols = [Symbol(name=name, kind=k) for k in kind]
        return self.lookup(symbols)

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
