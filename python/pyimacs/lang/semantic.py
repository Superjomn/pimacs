from pyimacs._C import ir

from . import core as pl


def add(a: pl.Value, b: pl.Value, builder: ir.builder) -> pl.Value:
    if a.dtype.is_float:
        return pl.Value(builder.create_fadd(a.handle, b.handle), a.dtype)
    elif a.dtype.is_int:
        return pl.Value(builder.create_iadd(a.handle, b.handle), a.dtype)
    else:
        raise NotImplementedError()
