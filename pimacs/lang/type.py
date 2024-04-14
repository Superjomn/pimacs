from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TypeId(Enum):
    INT = 1
    FLOAT = 2
    BOOL = 3
    STRING = 4
    CUSTOMED = 5


@dataclass
class TypeBase:
    __slots__ = ['type_id', 'name']
    type_id: TypeId
    name: Optional[str] = None

    def __str__(self):
        return self.name or self.type_id.name.lower()


@dataclass
class Type(TypeBase):
    __slots__ = ['type_id', 'name', 'inner_types']
    inner_types: Optional[List["Type"]] = None

    def __str__(self) -> str:
        if not self.inner_types:
            return super().__str__()
        return f"{self.name or self.type_id.name.lower()}[{', '.join(map(str, self.inner_types))}]"


# Built-in types
Int = Type(TypeId.INT)
Float = Type(TypeId.FLOAT)
Bool = Type(TypeId.BOOL)
Str = Type(TypeId.STRING)
Customed = Type(TypeId.CUSTOMED)
