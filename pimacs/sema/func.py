from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import pimacs.ast.type as _ty
from pimacs.ast.ast import CallParam, FuncCall, FuncDecl
from pimacs.ast.type import Type as _type

from .utils import Scoped


@dataclass(slots=True, frozen=True)
class FuncSymbol:
    """
    FuncSymbol represents a function symbol in the FuncTable.
    """

    name: str
    context: Tuple[Union["ModuleId", "ClassId"], ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.context:
            # check the ClassId is the last element if there is any
            if any(isinstance(x, ClassId) for x in self.context[:-1]):
                raise ValueError("ClassId should be the last element")


@dataclass(slots=True, unsafe_hash=True)
class FuncSig:
    """Function signature"""

    symbol: FuncSymbol
    input_types: Tuple[_ty.Type, ...]
    output_type: _ty.Type

    @classmethod
    def create(cls, func: "FuncDecl"):
        return_type = func.return_type if func.return_type else _ty.Nil
        symbol = FuncSymbol(func.name)
        # TODO: Support the context
        return FuncSig(
            symbol=symbol,
            input_types=tuple(arg.type for arg in func.args),
            output_type=return_type,
        )

    def match_call(self, args: List[CallParam]) -> bool:
        """Check if the arguments match the signature"""
        if len(args) != len(self.input_types):
            return False
        # TODO: process the variadic arguments
        for arg, ty in zip(args, self.input_types):
            if not arg.value.get_type().is_subtype(ty):
                return False
        return True


@dataclass(slots=True)
class FuncOverloads:
    """FuncOverloads represents the functions with the same name but different signatures.
    It could be records in the symbol table, and it could be scoped."""

    symbol: FuncSymbol
    # funcs with the same name
    funcs: Dict[FuncSig, FuncDecl] = field(default_factory=dict)

    def lookup(self, args: List[CallParam]) -> Optional[FuncDecl]:
        """Find the function that matches the arguments"""
        for sig in self.funcs:
            if sig.match_call(args):
                return self.funcs[sig]
        return None

    def add_func(self, func: FuncDecl):
        sig = FuncSig.create(func)
        assert sig not in self.funcs
        self.funcs[sig] = func

    def __add__(self, other: "FuncOverloads") -> "FuncOverloads":
        assert self.symbol == other.symbol
        assert self.funcs.keys().isdisjoint(other.funcs.keys())
        new_funcs = self.funcs.copy()
        new_funcs.update(other.funcs)
        return FuncOverloads(self.symbol, new_funcs)

    def __getitem__(self, sig: FuncSig) -> FuncDecl:
        return self.funcs[sig]

    def __contains__(self, sig: FuncSig) -> bool:
        return sig in self.funcs

    def __len__(self) -> int:
        return len(self.funcs)

    def __iter__(self):
        return iter(self.funcs.values())


@dataclass(slots=True)
class ModuleId:
    """ID for a module."""

    name: str


@dataclass(slots=True)
class ClassId:
    name: str


class FuncTable(Scoped):
    def __init__(self) -> None:
        self.scopes: List[Dict[FuncSymbol, FuncOverloads]] = [{}]  # global

    def lookup(self, symbol: FuncSymbol) -> Optional[FuncOverloads]:
        # get a FuncOverloads holding all the functions with the same symbol
        overloads: List[FuncOverloads] = []
        for scope in reversed(self.scopes):
            record = scope.get(symbol)
            if record:
                overloads.append(record)
        # TODO: optimize the performance
        if overloads:
            tmp = overloads.pop()
            for overload in overloads:
                tmp += overload
            return tmp
        return None

    def insert(self, func: FuncDecl):
        symbol = FuncSymbol(func.name)
        record = self.scopes[-1].get(symbol)
        if record:
            record.add_func(func)
        else:
            record = FuncOverloads(symbol)
            record.add_func(func)
            self.scopes[-1][symbol] = record

    def push_scope(self, kind: str = ""):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()
