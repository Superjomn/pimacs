from dataclasses import dataclass
from typing import *

import pyimacs.lang as pyl
from pyimacs.lang import ir
from pyimacs.lang.extension import Ext, builder, ctx, module, register_extern


@dataclass
class String(Ext):
    handle: object

    def __init__(self, x=None):
        if isinstance(x, String):
            self.handle = x.handle
        elif isinstance(x, str):
            self.handle = builder().create_string(x)
        elif isinstance(x, ir.Value):
            self.handle = x
        elif isinstance(x, pyl.Value):
            self.handle = x.handle
        else:
            raise NotImplementedError(f"Cannot create String from {x}")

    def __sizeof__(self) -> int:
        return length(self.handle)

    def __eq__(self, __o: Union["String", str]) -> bool:
        if isinstance(__o, str):
            other = String(__o)
            return string_eq(self.handle, other.handle)
        elif isinstance(__o, String):
            return string_eq(self.handle, __o.handle)
        assert NotImplementedError()

    def __ne__(self, __o: "String") -> bool:
        return not self.__eq__(__o)

    def __getitem__(self, idx: Any) -> str:
        assert isinstance(idx, slice)

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            return substring(self.handle, start, stop)

    def __add__(self, __o: "String") -> "String":
        return concat(self.handle, __o.handle)

    def __handle_return__(self):
        return self.handle


@register_extern("length")
def length(x: str) -> int: ...


@register_extern("substring")
def substring(x: str, start: int, end: int) -> str: ...


@register_extern("substring")
def substring_to_end(x: str, start: int) -> str: ...


@register_extern("concat")
def concat(a: str, b: str) -> str: ...


@register_extern("string-eq")
def string_eq(a: str, b: str) -> bool: ...
