import sys
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple, Union

from tabulate import tabulate  # type: ignore

from pimacs.ast import ast


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


@dataclass(unsafe_hash=True, slots=True)
class Symbol:
    """
    Represent any kind of symbol and is comparable.
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
    annotation: Optional[ast.Function.Annotation] = field(
        default=None, hash=True, repr=False)

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
    def __init__(self, name: str, context: Tuple[Union[ModuleId, ClassId], ...] = (), annotation: Optional[ast.Function.Annotation] = None):
        super().__init__(name, Symbol.Kind.Func, context=context, annotation=annotation)


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
