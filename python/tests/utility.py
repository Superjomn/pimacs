from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.lang.core import *


def build_mod0():
    ctx = ir.MLIRContext()
    ctx.load_pyimacs()
    builder = ir.Builder(ctx)
    mod = builder.create_module()
    int_ty = Int.to_ir(builder)
    func_ty = builder.get_function_ty([int_ty], [int_ty])
    func = builder.get_or_insert_function(mod, "add", func_ty, "public")
    mod.push_back(func)

    entry_block = func.add_entry_block()
    builder.set_insertion_point_to_start(entry_block)
    arg0 = func.args(0)
    eq0 = builder.create_add(arg0, builder.get_int32(32))
    builder.ret([eq0])
    return mod, ctx


def init_mod():
    ctx = ir.MLIRContext()
    ctx.load_pyimacs()
    builder = ir.Builder(ctx)
    mod = builder.create_module()
    return mod, builder, ctx
