from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union


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
    context: Tuple[Union[ModuleId, ClassId], ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.context:
            # check the ClassId is the last element if there is any
            if any(isinstance(x, ClassId) for x in self.context[:-1]):
                raise ValueError("ClassId should be the last element")

    def __str__(self):
        module_prefix = "::".join(map(str, self.context)) + "::" if self.context else ""
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
