from enum import Enum
from typing import *


class TypeKind(Enum):
    INT = 0
    FLOAT = 1
    BOOL = 2
    STRING = 3
    OBJECT = 4
    VOID = 5


class DataType:
    name2kind = {
        'int': TypeKind.INT,
        'float': TypeKind.FLOAT,
        'bool': TypeKind.BOOL,
        'string': TypeKind.STRING,
        'object': TypeKind.OBJECT,
        'void': TypeKind.VOID,
    }

    def __int__(self, name: str):
        self.dtype = DataType.name2kind.get(name)

    @property
    def is_int(self):
        return self.dtype == TypeKind.INT

    @property
    def is_string(self):
        return self.dtype == TypeKind.STRING

    @property
    def is_float(self):
        return self.dtype == TypeKind.FLOAT

    @property
    def is_object(self):
        return self.dtype == TypeKind.OBJECT


Void = DataType("void")
Int = DataType("int")
Float = DataType("float")
String = DataType("string")
Bool = DataType("bool")


class Value:
    def __init__(self, handle: Any, dtype: DataType):
        self.dtype = dtype
        self.handle = handle

    def __str__(self) -> str:
        return "<Value of " + str(self.dtype.dtype) + ">"


def to_value(x, builder):
    if isinstance(x, bool):
        return Value(builder.get_int1(x), Bool)
    if isinstance(x, int):
        return Value(builder.get_int32(x), Int)
    if isinstance(x, float):
        return Value(builder.get_float32(x), Float)
    if isinstance(x, Value):
        return x
