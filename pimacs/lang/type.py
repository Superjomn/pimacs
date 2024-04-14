from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TypeId(Enum):
    INT = 1
    FLOAT = 2
    BOOL = 3
    STRING = 4
    CUSTOMED = 5
    NIL = 6


@dataclass(slots=True)
class TypeBase:
    type_id: TypeId
    _name: Optional[str] = None

    def __str__(self):
        return self._name or self.type_id.name.lower()


@dataclass(slots=True)
class Type(TypeBase):
    inner_types: Optional[List["Type"]] = None

    def __str__(self) -> str:
        if not self.inner_types:
            return super().__str__()
        return f"{self._name or self.type_id.name.lower()}[{', '.join(map(str, self.inner_types))}]"


# Built-in types
Int = Type(TypeId.INT)
Float = Type(TypeId.FLOAT)
Bool = Type(TypeId.BOOL)
Str = Type(TypeId.STRING)
Customed = Type(TypeId.CUSTOMED)
Nil = Type(TypeId.NIL)

STR_TO_PRIMITIVE_TYPE = {
    'Int': Int,
    'Float': Float,
    'Bool': Bool,
    'Str': Str,
    'Customed': Customed,
    'Nil': Nil,
}


def parse_primitive_type(type_str: str) -> Optional[Type]:
    return STR_TO_PRIMITIVE_TYPE.get(type_str, None)
