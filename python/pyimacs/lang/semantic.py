from dataclasses import dataclass
from typing import *

from pyimacs._C.libpyimacs import pyimacs

from . import core as pl

ir = pyimacs.ir


class BinOp:
    op_builder_t = Tuple[Callable, Callable]
    add_ = (lambda builder: builder.create_fadd,
            lambda builder: builder.create_add)
    sub_ = (lambda builder: builder.create_fsub,
            lambda builder: builder.create_sub)
    mul_ = (lambda builder: builder.create_fmul,
            lambda builder: builder.create_mul)
    div_ = (lambda builder: builder.create_fdiv,
            lambda builder: builder.create_div)

    @staticmethod
    def bin_op(op_builder: op_builder_t, a: pl.Value, b: pl.Value, builder: ir.builder) -> pl.Value:
        if a.dtype.is_float:
            return pl.Value(op_builder[0](builder)(a.handle, b.handle), a.dtype)
        if a.dtype.is_int:
            return pl.Value(op_builder[1](builder)(a.handle, b.handle), a.dtype)
        raise NotImplementedError()


def add(a: pl.Value, b: pl.Value, builder: ir.builder) -> pl.Value:
    return BinOp.bin_op(BinOp.add_, a, b, builder)


def sub(a: pl.Value, b: pl.Value, builder: ir.builder) -> pl.Value:
    return BinOp.bin_op(BinOp.sub_, a, b, builder)


def div(a: pl.Value, b: pl.Value, builder: ir.builder) -> pl.Value:
    return BinOp.bin_op(BinOp.div_, a, b, builder)


def mul(a: pl.Value, b: pl.Value, builder: ir.builder) -> pl.Value:
    return BinOp.bin_op(BinOp.mul_, a, b, builder)


def constant(v: Any, builder: ir.builder):
    if type(v) is int:
        return pl.Value(builder.get_int32(v), pl.Int)
    if type(v) is float:
        return pl.Value(builder.get_float(v), pl.Float)
    raise NotImplementedError()
