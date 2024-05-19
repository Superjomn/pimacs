import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from tabulate import tabulate  # type: ignore


class ScopeKind(Enum):
    Local = 0
    Global = 1
    Class = 2
    Func = 3


@dataclass(slots=True)
class ModuleId:
    """ID for a module."""

    name: str


@dataclass(slots=True)
class ClassId:
    name: str


class Scoped:
    @abstractmethod
    def push_scope(self, kind: ScopeKind = ScopeKind.Local):
        pass

    @abstractmethod
    def pop_scope(self):
        pass

    @contextmanager
    def scope_guard(self, kind: ScopeKind = ScopeKind.Local):
        self.push_scope(kind)
        try:
            yield
        finally:
            self.pop_scope()


@dataclass(unsafe_hash=True, slots=True)
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
        TypeAlas = 6

        def __str__(self):
            return self.name

    name: str  # the name without "self." prefix if it is a member
    kind: Kind

    # The module or class of the symbol, it could be a chain of modules and classes.
    # `mod0::mod1::var0`
    # `mod0::class0::var0`
    context: Tuple[Union[ModuleId, ClassId], ...] = field(
        default_factory=tuple)

    def __post_init__(self):
        if self.context:
            # check the ClassId is the last element if there is any
            if any(isinstance(x, ClassId) for x in self.context[:-1]):
                raise ValueError("ClassId should be the last element")

    def __str__(self):
        module_prefix = "::".join(
            map(str, self.context)) + "::" if self.context else ""
        return f"{module_prefix}{self.kind.name}({self.name})"


class FuncSymbol(Symbol):
    def __init__(self, name: str, context: Tuple[Union[ModuleId, ClassId], ...] = ()):
        super().__init__(name, Symbol.Kind.Func, context=context)


class VarSymbol(Symbol):
    def __init__(self, name: str, context: Tuple[Union[ModuleId, ClassId], ...] = ()):
        super().__init__(name, Symbol.Kind.Var, context=context)


class ClassSymbol(Symbol):
    def __init__(self, name: str, context: Tuple[Union[ModuleId, ClassId], ...] = ()):
        super().__init__(name, Symbol.Kind.Class, context=context)


class MemberSymbol(Symbol):
    def __init__(self, name: str, context: Tuple[Union[ModuleId, ClassId], ...] = ()):
        super().__init__(name, Symbol.Kind.Member, context=context)


SymbolItem = Any


@dataclass
class Scope:
    data: Dict[Symbol, SymbolItem] = field(default_factory=dict)

    kind: ScopeKind = ScopeKind.Local

    def add(self, symbol: Symbol, item: SymbolItem):
        if symbol in self.data:
            raise KeyError(f"{item.loc}\nSymbol {symbol} already exists")
        self.data[symbol] = item

    def get(self, symbol: Symbol) -> SymbolItem | None:
        return self.data.get(symbol, None)

    def __contains__(self, symbol: Symbol):
        return symbol in self.data

    def print_summary(self):
        table = [["Symbol", "Kind", "Summary"]]
        for symbol, item in self.data.items():
            table.append([symbol.name, symbol.kind, str(item)[:50]])

        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


class bcolors(Enum):
    NONE = ""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_colored(msg: str, color: bcolors = bcolors.NONE):
    if color == bcolors.NONE:
        sys.stderr.write(msg)
    else:
        sys.stderr.write(f"{color.value}{msg}{bcolors.ENDC.value}")
