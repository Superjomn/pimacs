from dataclasses import dataclass
from typing import *

import pyimacs.lang as pyl
from pyimacs.lang import ir
from pyimacs.lang.extension import (Ext, arg_to_mlir, builder, ctx, module,
                                    register_extern)


def make_symbol(name: str):
    name = arg_to_mlir(name)
    return builder().make_symbol(name).get_result(0)
