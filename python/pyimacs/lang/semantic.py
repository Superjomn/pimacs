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
    def bin_op(op_builder: op_builder_t, a: pl.Value, b: pl.Value, builder: ir.Builder) -> pl.Value:
        if a.dtype.is_float:
            return pl.Value(op_builder[0](builder)(a.handle, b.handle), a.dtype)
        if a.dtype.is_int:
            return pl.Value(op_builder[1](builder)(a.handle, b.handle), a.dtype)
        raise NotImplementedError()


def add(a: pl.Value, b: pl.Value, builder: ir.Builder) -> pl.Value:
    return BinOp.bin_op(BinOp.add_, a, b, builder)


def sub(a: pl.Value, b: pl.Value, builder: ir.Builder) -> pl.Value:
    return BinOp.bin_op(BinOp.sub_, a, b, builder)


def div(a: pl.Value, b: pl.Value, builder: ir.Builder) -> pl.Value:
    return BinOp.bin_op(BinOp.div_, a, b, builder)


def mul(a: pl.Value, b: pl.Value, builder: ir.Builder) -> pl.Value:
    return BinOp.bin_op(BinOp.mul_, a, b, builder)


def constant(v: Any, builder: ir.Builder):
    if type(v) is int:
        return pl.Value(builder.get_int32(v), pl.Int)
    if type(v) is float:
        return pl.Value(builder.get_float(v), pl.Float)
    if type(v) is bool:
        return pl.Value(builder.get_int1(v), pl.Bool)
    if type(v) is str:
        return pl.Value(builder.get_string(v), pl.String)
    raise NotImplementedError()


def bitcast(input: pl.Value,
            dst_ty: pl.DataType,
            builder: ir.Builder) -> pl.Value:
    src_ty = input.dtype
    if src_ty == dst_ty:
        return input
    return pl.Value(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)),
                    dst_ty)


@dataclass(init=True,
           eq=True,
           repr=True,
           unsafe_hash=True)
class ExternalCallable:
    func_name: str
    return_type: pl.DataType
    attrs: Set[str]
    num_args: int = 0

    def __call__(self, *args, **kwargs):
        assert len(args) == self.num_args
        assert "builder" in kwargs
        for k in kwargs.keys():
            if k != "builder":
                assert k in self.attrs
        builder = kwargs['builder']

        args = [arg.handle if isinstance(
            arg, pl.Value) else arg for arg in args]
        print('args', args)
        return builder.extern_call(self.return_type.to_ir(builder), self.func_name, args)
