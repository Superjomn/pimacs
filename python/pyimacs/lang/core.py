from dataclasses import dataclass
from enum import Enum
from typing import *

from pyimacs._C.libpyimacs.pyimacs import ir


class TypeKind(Enum):
    INT = 0
    FLOAT = 1
    BOOL = 2
    STRING = 3
    OBJECT = 4
    VOID = 5


@dataclass(init=True,
           repr=True,
           eq=True,
           unsafe_hash=True,
           order=True)
class DataType:
    name2kind = {
        'int': TypeKind.INT,
        'float': TypeKind.FLOAT,
        'bool': TypeKind.BOOL,
        'string': TypeKind.STRING,
        'object': TypeKind.OBJECT,
        'void': TypeKind.VOID,
    }

    name: str

    @property
    def dtype(self):
        return DataType.name2kind.get(self.name)

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

    def to_ir(self, builder: ir.Builder) -> ir.Type:
        dic = {
            "void": builder.get_void_ty(),
            "int": builder.get_int32_ty(),
            "float": builder.get_double_ty(),
            "string": builder.get_string_ty(),
            "object": builder.get_object_ty(),
        }
        assert self.name in dic
        return dic[self.name]


Void = DataType("void")
Int = DataType("int")
Float = DataType("float")
String = DataType("string")
Object = DataType("object")
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


@dataclass(init=True, repr=True)
class FunctionType(DataType):
    ret_types: List[DataType]
    param_types: List[DataType]
    name: ClassVar[str] = "function_type"

    def to_ir(self, builder: ir.Builder):
        ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
        ret_types = [ret_type.to_ir(builder) for ret_type in self.ret_types]

        return builder.get_function_ty(ir_param_types, ret_types)
