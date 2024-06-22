import os
from typing import *

from tabulate import tabulate  # type: ignore

import pimacs.ast.ast as ast
import pimacs.ast.type as _ty

from .symbol_table import SymbolTable
from .utils import Symbol, SymbolItem, bcolors, print_colored

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
