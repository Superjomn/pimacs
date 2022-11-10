from enum import Enum
from typing import *


class TypeKind(Enum):
    INT = 0
    FLOAT = 1
    BOOL = 2
    STRING = 3
    OBJECT = 4


class DataType:
    def __int__(self, dtype: TypeKind):
        self.dtype = dtype

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


class Value:
    def __init__(self, handle: Any, dtype: DataType):
        self.dtype = dtype
        self.handle = handle

    def __str__(self) -> str:
        return "<Value of " + str(self.dtype.dtype) + ">"
