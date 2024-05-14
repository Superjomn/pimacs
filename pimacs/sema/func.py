from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pimacs.ast.type as _ty
from pimacs.ast.ast import CallParam, FuncCall, FuncDecl
from pimacs.ast.type import Type as _type


@dataclass(slots=True)
class FuncSig:
    """Function signature"""

    name: str
    input_types: List[_ty.Type]
    output_type: _ty.Type

    @classmethod
    def create(cls, func: "FuncDecl"):
        return_type = func.return_type if func.return_type else _ty.Nil
        return FuncSig(
            name=func.name,
            input_types=[arg.type for arg in func.args],
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

    name: str
    # funcs with the same name
    funcs: Dict[FuncSig, FuncDecl] = field(default_factory=dict)

    def lookup(self, args: List[CallParam]) -> Optional[FuncDecl]:
        """Find the function that matches the arguments"""
        for sig in self.funcs:
            if sig.match_call(args):
                return self.funcs[sig]
        return None
