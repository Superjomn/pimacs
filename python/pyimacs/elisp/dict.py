from dataclasses import dataclass
from typing import *

import pyimacs.lang as pyl
from pyimacs.lang import ir
from pyimacs.lang.extension import Ext, builder, ctx, module, register_extern


@dataclass
class Dict(Ext):
    handle: Any

    def __init__(self, x=None):
        if x is None:
            self.handle = ht_create()
        elif isinstance(x, Dict):
            self.handle = x.handle
        elif isinstance(x, dict):
            pass

    def __getitem__(self, key: str) -> Any:
        return ht_get(self.handle, key)

    def __setitem__(self, key: str, value: Any) -> None:
        return ht_set(self.handle, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return ht_get(self.handle, key)
        else:
            return default

    def keys(self) -> List[str]:
        return ht_keys(self.handle)

    def values(self) -> List[Any]:
        return ht_values(self.handle)

    def __len__(self) -> int:
        return ht_size(self.handle)


@register_extern("ht-create")
def ht_create() -> object: ...


@register_extern("ht-get")
def ht_get(table: object, key: str) -> object: ...


@register_extern("ht-set!")
def ht_set(table: object, key: str, value: object) -> None: ...


@register_extern("ht-keys")
def ht_keys(table: object) -> List[str]: ...


@register_extern("ht-values")
def ht_values(table: object) -> List[Any]: ...


@register_extern("ht-size")
def ht_size(table: object) -> int: ...
