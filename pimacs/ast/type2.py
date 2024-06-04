from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


class Type:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = (cls, args, tuple(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, name, parent=None, params: Optional[Tuple["Type", ...]] = None, is_concrete=True):
        self.name = name
        self.params = params or tuple()
        self.parent = parent
        self._is_concrete = is_concrete
        self._initialized = True

    @property
    def is_concrete(self) -> bool:
        return self._is_concrete


class BasicType(Type):
    def __init__(self, name, parent=None):
        super().__init__(name=name, parent=parent, is_concrete=True)


class CompositeType(Type):
    def __init__(self, name, parent=None, params: Optional[List[Type]] = None):
        super().__init__(name, parent, params=params)

    @property
    def is_concrete(self):
        return all(param.is_concrete for param in self.params)


class PlaceholderType(Type):
    def __init__(self, name, parent=None):
        super().__init__(name, parent, is_concrete=False)

    def __repr__(self):
        return self.name

    def compatible_with(self, other):
        return True


class GenericType(Type):
    def __init__(self, name, parent=None, *params):
        super().__init__(name, parent, params=params)


class FunctionType(Type):
    def __init__(self, return_type, arg_types: List[Type], params: Optional[List[Type]] = None):
        super().__init__("Function", params=params)
        self.return_type = return_type
        self.arg_types = arg_types

    def __repr__(self):
        arg_types = ', '.join(repr(arg) for arg in self.arg_types)
        param_types = ', '.join(repr(param) for param in self.params)
        return f"({arg_types})[{param_types}] -> {self.return_type}"

# Basic types


class NumberType(BasicType):
    def __init__(self, parent=None):
        super().__init__(name="Number", parent=parent)


class IntType(BasicType):
    def __init__(self):
        super().__init__("Int", parent=NumberType())


class FloatType(BasicType):
    def __init__(self):
        super().__init__("Float", parent=NumberType())


class BoolType(BasicType):
    def __init__(self):
        super().__init__("Bool")


class StrType(BasicType):
    def __init__(self):
        super().__init__("Str")


# basic types
Number = NumberType()
Int = IntType()
Float = FloatType()
Bool = BoolType()
Str = StrType()
