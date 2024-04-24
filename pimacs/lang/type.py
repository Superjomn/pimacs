from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TypeId(Enum):
    INT = 'Int'
    FLOAT = 'Float'
    BOOL = 'Bool'
    STRING = 'Str'
    CUSTOMED = 'Customed'
    Unk = 'Unk'
    NIL = 'nil'


@dataclass(slots=True)
class TypeBase:
    type_id: TypeId
    _name: Optional[str] = None

    def __str__(self):
        return self._name or self.type_id.value


@dataclass(slots=True)
class Type(TypeBase):
    inner_types: Optional[List["Type"]] = None

    def __str__(self) -> str:
        if not self.inner_types:
            return self._name or self.type_id.value
        return f"{self._name or self.type_id.name.lower()}[{', '.join(map(str, self.inner_types))}]"


# Built-in types
Int = Type(TypeId.INT)
Float = Type(TypeId.FLOAT)
Bool = Type(TypeId.BOOL)
Str = Type(TypeId.STRING)
Customed = Type(TypeId.CUSTOMED)
Nil = Type(TypeId.NIL)
# Lisp type is a special type that is used to represent the type of a lisp object
LispType = Type(TypeId.CUSTOMED, '__LispObject__')
Unk = Type(TypeId.Unk)

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
