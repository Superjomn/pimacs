from typing import *

import pyimacs.lang as pyl
import pytest


@pytest.mark.parametrize('type_name',
                         [
                             "float",
                             "string",
                             "int",
                             "bool",
                         ])
def test_DataType(type_name: str):
    builder, _ = create_builder()

    float_ty = pyl.DataType("float")
    ty = float_ty.to_ir(builder)
    print(ty)


@pytest.mark.parametrize('value', [
    123,
    1.23,
    True,
    False,
    "string",
])
def test_to_Value(value: Any):
    builder, _ = create_builder()
    pyl.to_value(value, builder)


def create_builder() -> pyl.ir.builder:
    ctx = pyl.ir.context()
    ctx.load_pyimacs()
    builder = pyl.ir.Builder(ctx)
    return builder, ctx
