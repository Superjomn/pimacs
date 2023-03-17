from dataclasses import dataclass
from typing import *

import pyimacs.lang as pyl
from pyimacs.elisp.core import *
from pyimacs.lang import ir
from pyimacs.lang.extension import Ext, builder, ctx, module, register_extern


@dataclass
class List:
    handle: Any

    def __eq__(self, other):
        return eq(self.handle, other.handle)

    def __len__(self):
        return length(self.handle)

    def append(self, x: object) -> None:
        return _append(x, self.handle)

    def push(self, x: object) -> None:
        return _push(x, self.handle)

    def __getitem__(self, n: int) -> object:
        return _nth(n, self.handle)


@register_extern("append")
def _append(item: object, lst: object) -> object: ...


@register_extern("push")
def _push(item: object, lst: object) -> object: ...


@register_extern("nth")
def _nth(n: int, lst: object) -> object: ...
