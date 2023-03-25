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


class Guard(Ext):
    def __init__(self, name: str, *args):
        name = _trans_arg_to_mlir(name)
        args = [_trans_arg_to_mlir(x) for x in args]
        self.name = name
        self.guard_op = builder().guard(self.name, args)

    def __enter__(self):
        self.pre = builder().get_insertion_point()
        builder().set_insertion_point_to_start(self.guard_op.get_body_block())

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        builder().restore_insertion_point(self.pre)


@register_extern("load")
def load(path: str) -> None: ...


@register_extern("require")
def _require(feature: str, *args) -> None: ...


@register_extern("setq")
def _setq(*args): ...

def cl_assert(condition: bool, message: str="") -> None:
    _cl_assert(condition, message)

# TODO[Superjomn]: replace with make_symbol?


@register_extern("quote")
def quote(x: object) -> object: ...


@register_extern("eq")
def eq(x: object, y: object) -> bool: ...


@register_extern("length")
def length(x: object) -> int: ...


@register_extern("cl-assert")
def _cl_assert(condition: bool, message: str) -> None: ...

def _trans_arg_to_mlir(arg):
    if isinstance(arg, pyl.core.Value):
        return arg.handle
    if isinstance(arg, Ext):
        return arg._handle
    if isinstance(arg, int):
        return builder().get_int(arg)
    if isinstance(arg, str):
        return builder().get_string(arg)
    if isinstance(arg, bool):
        return builder().get_bool(arg)
    return arg