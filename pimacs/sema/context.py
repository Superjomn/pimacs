import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import *

from tabulate import tabulate  # type: ignore

import pimacs.ast.ast as ast
import pimacs.ast.type as _ty

from .symbol_table import SymbolTable
from .utils import (ClassId, ModuleId, Scoped, ScopeKind, Symbol, SymbolItem,
                    bcolors, print_colored)

PIMACS_SEMA_RAISE_EXCEPTION: bool = os.environ.get(
    "PIMACS_SEMA_RAISE_EXCEPTION", "0") == "1"


class SemaError(Exception):
    pass


class ModuleContext:
    """Context is a class that represents the context of the Module.
    The states includs
        - function symbols
        - variable symbols
        - type symbols
        - class symbols

    The module context could be nested.
    """

    def __init__(self, name: str = "main", enable_exception: bool = PIMACS_SEMA_RAISE_EXCEPTION):
        self._name = name
        self.symbols = SymbolTable()
        self._enable_exception = enable_exception

    def get_symbol(
        self, symbol: Symbol
    ) -> Optional[SymbolItem]:
        return self.symbols.lookup(symbol)

    # TODO: Refine the type system
    def get_type(
        self, name: str, subtypes: Optional[Tuple[_ty.Type, ...]] = None
    ) -> Optional[_ty.Type]:
        key = f"{name}[{', '.join(map(str, subtypes))}]" if subtypes else name
        raise NotImplementedError()
        return None

    def symbol_exists(self, symbol: Symbol) -> bool:
        return self.symbols.lookup(symbol) is not None

    def report_sema_error(self, node: ast.Node, message: str):
        if self._enable_exception:
            raise SemaError(f"{node.loc}\nError: {message}")
        else:
            node.sema_failed = True
            print_colored(f"{node.loc}\n")
            print_colored(f"Error: {message}\n\n", bcolors.FAIL)

    @property
    def name(self) -> str:
        """Module name."""
        return self._name


'''
class TypeSystem(_ty.TypeSystemBase):
    """
    The TypeSystem to handle the types and classes.
    """

    def __init__(self, symtbl: SymbolTable) -> None:
        super().__init__()
        # symtbl holds the typealias, class symbols
        # TODO: Consider modules
        self.symtbl = symtbl

    def add_type_alias(self, name: str, ty: _ty.Type) -> None:
        # Type alias is similar to variable, it should be added to the current scope
        self.symtbl.insert(Symbol(name, Symbol.Kind.TypeAlas), ty)

    def get_type(self, name: str) -> Optional[_ty.Type]:
        # Shadow the builtin Types is not allowed, so it is safe to get the type alias first
        if symbol := self.symtbl.lookup(name=name, kind=Symbol.Kind.TypeAlas):
            return symbol
        return super().get_type(name)

    def get_types_by_name(self, name: str) -> Iterable[_ty.Type]:
        return filter(lambda ty: ty.name == name, [ty for k, ty in self.types.items()])

    def get_ListType(self, inner_type: _ty.Type) -> _ty.Type:
        return self.define_composite_type("List", inner_type)

    def get_SetType(self, inner_type: _ty.Type) -> _ty.Type:
        return self.define_composite_type("Set", inner_type)

    def get_DictType(self, key_type: _ty.Type, value_type: _ty.Type) -> _ty.Type:
        return self.define_composite_type("Dict", key_type, value_type)

    def get_CustomType(self, name: str, *args: _ty.Type) -> _ty.Type:
        return self.define_composite_type(name, *args)

    type_place_holder_counter = 0

    def get_type_placeholder(self, name: str) -> _ty.TemplateType:
        """
        Get a placeholer for type, such T in List[T].
        """
        return _ty.TemplateType(name=name)

    def get_unique_type_placeholder(self) -> _ty.TemplateType:
        """
        Get a globally unique placeholder for type.
        """
        self.type_place_holder_counter += 1
        return self.get_type_placeholder(f"__T{self.type_place_holder_counter}")
'''
