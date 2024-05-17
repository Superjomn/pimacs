from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import pimacs.ast.type as _ty
from pimacs.ast.ast import CallParam, Expr, FuncCall, FuncDecl
from pimacs.ast.type import Type as _type

from .utils import FuncSymbol, Scoped, ScopeKind, Symbol


@dataclass(slots=True, unsafe_hash=True)
class FuncSig:
    """Function signature"""

    symbol: FuncSymbol
    input_types: Tuple[Tuple[str, _ty.Type], ...]
    output_type: _ty.Type

    @classmethod
    def create(cls, func: "FuncDecl"):
        return_type = func.return_type if func.return_type else _ty.Nil
        symbol = FuncSymbol(func.name)
        # TODO: Support the context
        return FuncSig(
            symbol=symbol,
            input_types=tuple((arg.name, arg.type) for arg in func.args),
            output_type=return_type,
        )

    # TODO: replace the List[CallParam] with FuncCall to support overloading based on return_type
    def match_call(self, params: List[CallParam | Expr]) -> bool:
        """Check if the arguments match the signature"""
        if len(params) != len(self.input_types):
            return False
        # optimize the performance
        args = {arg: type for arg, type in self.input_types}

        # TODO: Add the basic func-call rule back to FuncCall
        for no, param in enumerate(params):
            if isinstance(param, Expr):
                if not param.get_type().is_subtype(args[self.input_types[no][0]]):
                    return False
            else:
                if param.name not in args:
                    return False
                if not param.value.get_type().is_subtype(args[param.name]):
                    return False
        # TODO: process the variadic arguments
        return True


@dataclass(slots=True)
class FuncOverloads:
    """FuncOverloads represents the functions with the same name but different signatures.
    It could be records in the symbol table, and it could be scoped."""

    symbol: FuncSymbol
    # funcs with the same name
    funcs: Dict[FuncSig, FuncDecl] = field(default_factory=dict)

    def lookup(self, args: List[CallParam | Expr]) -> Optional[FuncDecl]:
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

    def contains(self, symbol: FuncSymbol) -> bool:
        return bool(self.lookup(symbol))

    def contains_locally(self, symbol: FuncSymbol) -> bool:
        return symbol in self.scopes[-1]

    def push_scope(self, kind: ScopeKind = ScopeKind.Local):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()
