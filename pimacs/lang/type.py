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
class TypeBase:
    type_id: TypeId
    name: Optional[str] = None

    def __str__(self):
        return self.name or self.type_id.value


@dataclass(slots=True, unsafe_hash=True)
class Type(TypeBase):
    inner_types: Tuple["Type", ...] = field(default_factory=tuple)
    is_optional: bool = False  # Optional type such as Int?

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


STR_TO_PRIMITIVE_TYPE = {
    "Int": Int,
    "Float": Float,
    "Bool": Bool,
    "Str": Str,
    "Nil": Nil,
}


def parse_primitive_type(type_str: str) -> Optional[Type]:
    return STR_TO_PRIMITIVE_TYPE.get(type_str, None)
