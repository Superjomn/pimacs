from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import *

import pimacs.ast.ast as ast
import pimacs.ast.type as _ty

from .func import FuncOverloads, FuncSig, FuncSymbol, FuncTable
from .utils import ClassId, ModuleId, Scoped, ScopeKind, Symbol


class ModuleContext:
    """Context is a class that represents the context of the Module.
    The states includs
        - function symbols
        - variable symbols
        - type symbols
        - class symbols

    The module context could be nested.
    """

    def __init__(self, name: str):
        self._name = name
        self._functions: Dict[str, ast.FuncDecl] = {}
        self._variables: Dict[str, ast.VarDecl] = {}
        self._classes: Dict[str, ast.ClassDef] = {}
        self._types: Dict[str, _ty.Type] = {}

    def get_symbol(
        self, name: str
    ) -> Optional[Union[ast.FuncDecl, ast.VarDecl, ast.ClassDef]]:
        if name in self._functions:
            return self._functions[name]
        if name in self._variables:
            return self._variables[name]
        if name in self._classes:
            return self._classes[name]
        return None

    def get_type(
        self, name: str, subtypes: Optional[Tuple[_ty.Type, ...]] = None
    ) -> Optional[_ty.Type]:
        key = f"{name}[{', '.join(map(str, subtypes))}]" if subtypes else name
        if key in self._types:
            return self._types[key]
        new_type = _ty.make_customed(name, subtypes)
        self._types[key] = new_type
        return new_type

    def symbol_exists(self, name: str) -> bool:
        return (
            name in self._functions or name in self._variables or name in self._classes
        )

    def add_function(self, func: ast.FuncDecl):
        self._functions[func.name] = func

    def add_variable(self, var: ast.VarDecl):
        self._variables[var.name] = var

    def add_class(self, cls: ast.ClassDef):
        self._classes[cls.name] = cls

    def get_function(self, name: str) -> Optional[ast.FuncDecl]:
        return self._functions.get(name)

    def get_variable(self, name: str) -> Optional[ast.VarDecl]:
        return self._variables.get(name)

    def get_class(self, name: str) -> Optional[ast.ClassDef]:
        return self._classes.get(name)

    @property
    def name(self) -> str:
        """Module name."""
        return self._name


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


class SymbolTable(Scoped):
    def __init__(self):
        self.scopes = [Scope(kind=ScopeKind.Global)]
        self.func_table = FuncTable()

    def push_scope(self, kind: ScopeKind = ScopeKind.Local):
        self.scopes.append(Scope(kind=kind))
        self.func_table.push_scope(kind=kind)

    def pop_scope(self):
        self.scopes.pop()
        self.func_table.pop_scope()

    def _add_function(self, func: ast.FuncDecl):
        self.func_table.insert(func)

    def insert(self, symbol: Symbol, item: SymbolItem):
        if symbol.kind == Symbol.Kind.Func:
            self._add_function(item)
        else:
            self.scopes[-1].add(symbol=symbol, item=item)
        return item

    def get_symbol(
        self,
        symbol: Optional[Symbol] = None,
        name: Optional[str] = None,
        kind: Optional[Symbol.Kind | List[Symbol.Kind]] = None,
    ) -> Optional[SymbolItem]:
        symbols = {symbol}
        if not symbol:
            assert name and kind
            symbols = (
                {Symbol(name=name, kind=kind)}
                if isinstance(kind, Symbol.Kind)
                else {Symbol(name=name, kind=k) for k in kind}
            )

        for symbol in symbols:
            for scope in reversed(self.scopes):
                ret = scope.get(symbol)
                if ret:
                    return ret
        return None

    def get_function(self, symbol: FuncSymbol) -> Optional[FuncOverloads]:
        return self.func_table.lookup(symbol)

    def contains(self, symbol: Symbol) -> bool:
        if symbol.kind is Symbol.Kind.Func:
            return self.func_table.contains(symbol)
        return any(symbol in scope for scope in reversed(self.scopes))

    def contains_locally(self, symbol: Symbol) -> bool:
        if symbol.kind is Symbol.Kind.Func:
            return self.func_table.contains_locally(symbol)
        return self.scopes[-1].get(symbol) is not None

    @property
    def current_scope(self):
        return self.scopes[-1]

    @contextmanager
    def scope_guard(self, kind=ScopeKind.Local):
        self.push_scope(kind)
        try:
            yield
        finally:
            self.pop_scope()


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
        symbol = self.symtbl.get_symbol(name=name, kind=Symbol.Kind.TypeAlas)
        if symbol:
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
