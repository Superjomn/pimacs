from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from multimethod import multimethod

import pimacs.ast.type as _ty
from pimacs.ast.ast import Call, CallParam, Expr, Function
from pimacs.ast.type import Type as _type

from .utils import (FuncSymbol, Scoped, ScopeKind, Symbol, SymbolItem,
                    print_colored)


class FuncDuplicationError(Exception):
    pass


@dataclass(slots=True, unsafe_hash=True)
class FuncSig:
    """Function signature"""

    symbol: FuncSymbol
    input_types: Tuple[Tuple[str, _ty.Type], ...]
    output_type: _ty.Type

    @classmethod
    def create(cls, func: "Function"):
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

    symbol: Symbol
    # funcs with the same name
    funcs: Dict[FuncSig, Function] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.symbol.name

    @multimethod
    def lookup(self, args: tuple) -> Optional[Function]:
        """Find the function that matches the arguments"""
        for sig in self.funcs:
            if sig.match_call(args):
                return self.funcs[sig]
        return None

    @multimethod  # type: ignore
    def lookup(self, sig: FuncSig) -> Optional[Function]:
        """Find the function that matches the signature"""
        return self.funcs.get(sig, None)

    def insert(self, func: Function):
        sig = FuncSig.create(func)
        if sig in self.funcs:
            raise FuncDuplicationError(f"Function {func.name} already exists")
        self.funcs[sig] = func

    def __iter__(self):
        return iter(self.funcs.values())

    def __add__(self, other: "FuncOverloads") -> "FuncOverloads":
        assert self.symbol == other.symbol
        assert self.funcs.keys().isdisjoint(other.funcs.keys())
        new_funcs = self.funcs.copy()
        new_funcs.update(other.funcs)
        return FuncOverloads(self.symbol, new_funcs)

    def __getitem__(self, sig: FuncSig) -> Function:
        return self.funcs[sig]

    def __contains__(self, sig: FuncSig) -> bool:
        return sig in self.funcs

    def __len__(self) -> int:
        return len(self.funcs)

    def __iter__(self):  # type: ignore
        return iter(self.funcs.values())

    def __repr__(self):
        return f"FuncOverloads[{self.symbol} x {len(self.funcs)}]"
