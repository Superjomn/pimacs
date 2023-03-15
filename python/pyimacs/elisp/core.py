from dataclasses import dataclass
from typing import *

import pyimacs.lang as pyl
from pyimacs.lang import ir
from pyimacs.lang.extension import (Ext, arg_to_mlir, builder, ctx, module,
                                    register_extern)


def make_symbol(name: str, is_keyword: bool = False):
    '''
    is_keyword: whether to prefix a ':' and get `:symbol`
    '''
    name = arg_to_mlir(name)
    return builder().make_symbol(name, is_keyword).get_result(0)


@register_extern("make_var_args")
def make_var_args(args: object) -> object:
    ''' Mark a tuple as var args, and it will be expanded in the arglist during codegen '''
    ...


def use_package(module: str, ensure: bool = True):
    _use_package(module)


@register_extern("setq")
def setq(name: str, value: object) -> None: ...


@register_extern("load")
def load(path: str) -> None: ...


def _use_package(module: str) -> None: ...
