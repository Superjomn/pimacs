from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TypeId(Enum):
    INT = "Int"
    FLOAT = "Float"
    BOOL = "Bool"
    STRING = "Str"
    Unk = "Unk"

    # coumpound types
    Set = "Set"
    Dict = "Dict"
    List = "List"

    NIL = "nil"

    CUSTOMED = "Customed"


@dataclass(slots=True)
class TypeBase(ABC):
    type_id: TypeId
    name: Optional[str] = None

    @property
    @abstractmethod
    def is_concrete(self) -> bool:
        # the type is concrete
        # i.e. a TypeTemplate is not concrete, such as List[T]
        return True

    def __str__(self):
        return self.name or self.type_id.value


@dataclass(slots=True, unsafe_hash=True)
class Type(TypeBase):
    inner_types: Tuple["Type", ...] = field(default_factory=tuple)
    is_optional: bool = False  # Optional type such as Int?

    @property
    def is_concrete(self) -> bool:
        return all(t.is_concrete for t in self.inner_types)

    def __post_init__(self):
        if self.is_optional:
            assert self.type_id != TypeId.NIL, "nil type cannot be optional"
        if self.type_id in (TypeId.Set, TypeId.List):
            assert (
                len(self.inner_types) == 1
            ), f"Set/List type must have exactly one inner type, got {len(self.inner_types)}"
        if self.type_id == TypeId.Dict:
            assert (
                len(self.inner_types) == 2
            ), f"Dict type must have exactly two inner types, got {len(self.inner_types)}"
        if self.type_id == TypeId.NIL:
            assert not self.inner_types, "nil type cannot have inner types"
        if self.type_id == TypeId.CUSTOMED:
            assert self.name is not None, "Customed type must have a name"
        if self.type_id == TypeId.Unk:
            assert not self.inner_types, "Unk type cannot have inner types"

    def __str__(self) -> str:
        if self.type_id is TypeId.NIL:
            return "nil"
        elif self.type_id is TypeId.Set:
            return "{%s}" % self.inner_types[0]
        elif self.type_id is TypeId.List:
            return "[%s]" % self.inner_types[0]
        elif self.type_id is TypeId.Dict:
            return "{%s: %s}" % self.inner_types
        elif self.type_id is TypeId.Unk:
            return "Unk"
        elif self.type_id is TypeId.INT:
            return "Int"
        elif self.type_id is TypeId.FLOAT:
            return "Float"
        elif self.type_id is TypeId.BOOL:
            return "Bool"
        elif self.type_id is TypeId.STRING:
            return "Str"
        elif self.type_id is TypeId.CUSTOMED:
            if not self.inner_types:
                return self.name or self.type_id.value
            optional = "?" if self.is_optional else ""

            return f"{self.name or self.type_id.name.lower()}[{', '.join(map(str, self.inner_types))}]{optional}"
        else:
            raise ValueError(f"Unknown type {self.type_id}")
        return ""

    def __repr__(self) -> str:
        return str(self)


@dataclass(slots=True)
class TemplateType(Type):
    """
    Template type, such as T.
    Such type should be substituted with a concrete type before type checking.

    TemplateType should be unique accross a whole module for easier to substitute with concrete types.
    """

    type_id: TypeId = field(default=TypeId.CUSTOMED, init=False)

    @property
    def is_concrete(self) -> bool:
        return False


@dataclass(slots=True)
class TypeTemplate(Type):
    """
    The type template, such as List in List[T].

    To declare a type template, use TypeTemplate instead of actual Type.
    e.g.

    # It will declare a class MyClass with two type parameters T0 and T1
    @template[T0, T1]
    class MyClass: ...

    The type MyClass[T0, T1] is a TypeTemplate, and it is not concrete.
    """

    def __post_init__(self):
        self.is_concrete = all(t.is_concrete for t in self.inner_types)


# Built-in types
Int = Type(TypeId.INT)
Float = Type(TypeId.FLOAT)
Bool = Type(TypeId.BOOL)
Str = Type(TypeId.STRING)
Nil = Type(TypeId.NIL)
# Lisp type is a special type that is used to represent the type of a lisp object
LispType = Type(TypeId.CUSTOMED, "Lisp")
Unk = Type(TypeId.Unk)


@dataclass(slots=True)
class SetType(Type):
    type_id: TypeId = field(default=TypeId.Set, init=False)


@dataclass(slots=True)
class ListType(Type):
    type_id: TypeId = field(default=TypeId.List, init=False)


@dataclass(slots=True)
class DictType(Type):
    type_id: TypeId = field(default=TypeId.Dict, init=False)
    key_type: Type | None = None
    value_type: Type | None = None

    def __post_init__(self):
        self.inner_types = (self.key_type, self.value_type)

    def __str__(self) -> str:
        assert self.key_type is not None
        assert self.value_type is not None
        return f"Dict[{self.key_type}, {self.value_type}]"


def make_customed(name: str, subtypes: Optional[Tuple["Type", ...]] = None) -> Type:
    type = Type(TypeId.CUSTOMED, name)
    if subtypes:
        type.inner_types = tuple(subtypes)
    return type


class TypeSystemBase:
    def __init__(self) -> None:
        self.types: Dict[str, Type] = {}

    def _init_basic_type(self):
        self.types["Int"] = Int
        self.types["Float"] = Float
        self.types["Bool"] = Bool
        self.types["Str"] = Str
        self.types["nil"] = Nil

    def define_composite_type(self, name: str, *args: Type) -> Type:
        ty = make_customed(name, args)
        key = str(ty)
        if key not in self.types:
            self.types[key] = make_customed(name, args)
        return self.types[key]

    def get_type(self, name: str) -> Optional[Type]:
        return self.types.get(name, None)


STR_TO_PRIMITIVE_TYPE = {
    "Int": Int,
    "Float": Float,
    "Bool": Bool,
    "Str": Str,
    "Nil": Nil,
}


def parse_primitive_type(type_str: str) -> Optional[Type]:
    return STR_TO_PRIMITIVE_TYPE.get(type_str, None)
