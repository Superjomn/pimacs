
from dataclasses import dataclass, field
from typing import *

import pyimacs.lang as pyl
from pyimacs.lang import ir
from pyimacs.lang.extension import (Ext, arg_to_mlir, builder, ctx, module,
                                    register_extern)


def _make_tuple(args: Tuple[Any]) -> object:
    assert args, "Tuple must have at least one element."
    print('make_tuple', arg_to_mlir(args))
    return builder().make_tuple(arg_to_mlir(args)).get_result(0)
