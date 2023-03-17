from dataclasses import dataclass
from typing import *

import pyimacs.lang as pyl
from pyimacs.lang import ir
from pyimacs.lang.extension import (Ext, arg_to_mlir, builder, ctx, module,
                                    register_extern)

__all__ = ("setq",
           "require",
           "make_symbol",
           "quote",
           "length",
           "eq",
           )


def setq(*args):
    '''
    (setq a value)
    '''
    assert args and len(args) % 2 == 0
    _setq(*args)


def require(feature: str, FILENAME=None, NOERROR=None) -> None:
    '''
    (require 'feature)
    '''
    args = []
    if FILENAME:
        args.append(FILENAME)
        if NOERROR:
            args.append(NOERROR)
    assert len(args) <= 2
    _require(feature, *args)


def make_symbol(name: str, is_keyword: bool = False):
    '''
    make a symbol
    '''
    name = arg_to_mlir(name)
    return builder().make_symbol(name, is_keyword).get_result(0)


@register_extern("load")
def load(path: str) -> None: ...


@register_extern("require")
def _require(feature: str, *args) -> None: ...


@register_extern("setq")
def _setq(*args): ...

# TODO[Superjomn]: replace with make_symbol?


@register_extern("quote")
def quote(x: object) -> object: ...


@register_extern("eq")
def eq(x: object, y: object) -> bool: ...


@register_extern("length")
def length(x: object) -> int: ...
