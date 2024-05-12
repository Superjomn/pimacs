from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import *

import pimacs.lang.ir as ir
import pimacs.lang.type as _ty


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
        self._functions: Dict[str, ir.FuncDecl] = {}
        self._variables: Dict[str, ir.VarDecl] = {}
        self._classes: Dict[str, ir.ClassDef] = {}
        self._types: Dict[str, _ty.Type] = {}

    def get_symbol(
        self, name: str
    ) -> Optional[Union[ir.FuncDecl, ir.VarDecl, ir.ClassDef]]:
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

    def add_function(self, func: ir.FuncDecl):
        self._functions[func.name] = func

    def add_variable(self, var: ir.VarDecl):
        self._variables[var.name] = var

    def add_class(self, cls: ir.ClassDef):
        self._classes[cls.name] = cls

    def get_function(self, name: str) -> Optional[ir.FuncDecl]:
        return self._functions.get(name)

    def get_variable(self, name: str) -> Optional[ir.VarDecl]:
        return self._variables.get(name)

    def get_class(self, name: str) -> Optional[ir.ClassDef]:
        return self._classes.get(name)

    @property
    def name(self) -> str:
        """Module name."""
        return self._name


@dataclass(unsafe_hash=True)
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

    def __str__(self):
        return f"{self.kind.name}({self.name})"


@dataclass
class ModuleId:
    ids: Tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        return "::".join(self.ids)


SymbolItem = Any


@dataclass
class Scope:
    data: Dict[Symbol, SymbolItem] = field(default_factory=dict)

    class Kind(Enum):
        Local = 0
        Global = 1
        Class = 2
        Func = 3

    kind: Kind = Kind.Local

    def add(self, symbol: Symbol, item: SymbolItem):
        if symbol in self.data:
            raise KeyError(f"{item.loc}\nSymbol {symbol} already exists")
        self.data[symbol] = item

    def get(self, symbol: Symbol) -> SymbolItem | None:
        return self.data.get(symbol, None)


class SymbolTable:
    def __init__(self):
        self.scopes = [Scope(kind=Scope.Kind.Global)]

    def push_scope(self, kind: Scope.Kind):
        self.scopes.append(Scope(kind=kind))

    def pop_scope(self):
        self.scopes.pop()

    def add_symbol(self, symbol: Symbol, item: SymbolItem):
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

    def contains(self, symbol: Symbol) -> bool:
        return any(symbol in scope for scope in reversed(self.scopes))

    def contains_locally(self, symbol: Symbol) -> bool:
        return self.scopes[-1].get(symbol) is not None

    @property
    def current_scope(self):
        return self.scopes[-1]

    @contextmanager
    def scope_guard(self, kind=Scope.Kind.Local):
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
        self.symtbl.add_symbol(Symbol(name, Symbol.Kind.TypeAlas), ty)

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
