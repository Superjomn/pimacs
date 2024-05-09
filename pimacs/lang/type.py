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

    def __str__(self) -> str:
        if not self.inner_types:
            return self.name or self.type_id.value
        return f"{self.name or self.type_id.name.lower()}[{', '.join(map(str, self.inner_types))}]"


# Built-in types
Int = Type(TypeId.INT)
Float = Type(TypeId.FLOAT)
Bool = Type(TypeId.BOOL)
Str = Type(TypeId.STRING)
Nil = Type(TypeId.NIL)
Customed = Type(TypeId.CUSTOMED)
# Lisp type is a special type that is used to represent the type of a lisp object
LispType = Type(TypeId.CUSTOMED, "__LispObject__")
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

    def __str__(self) -> str:
        assert self.key_type is not None
        assert self.value_type is not None
        return f"Dict[{self.key_type}, {self.value_type}]"


def make_customed(name: str) -> Type:
    return Type(TypeId.CUSTOMED, name)


STR_TO_PRIMITIVE_TYPE = {
    "Int": Int,
    "Float": Float,
    "Bool": Bool,
    "Str": Str,
    "Customed": Customed,
    "Nil": Nil,
}


def parse_primitive_type(type_str: str) -> Optional[Type]:
    return STR_TO_PRIMITIVE_TYPE.get(type_str, None)
