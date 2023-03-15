from pyimacs._C.libpyimacs.pyimacs import ir
from pyimacs.lang.core import *

from .utility import build_mod0, init_mod


def test_mod_get_function():
    mod, ctx = build_mod0()
    hello_fn = mod.get_llvm_function("add")
    print(hello_fn)
    print('body region', hello_fn.body())
    assert hello_fn


def test_mod_get_function_names():
    mod, ctx = build_mod0()
    funcs = mod.get_function_names()
    assert "add" in funcs


def test_function_get_body():
    mod, ctx = build_mod0()
    hello_fn = mod.get_llvm_function("add")
    region = hello_fn.body()
    assert region.size() == 1
    block = region.blocks(0)
    assert block.get_num_arguments() == 1
    ops = block.operations()
    assert len(ops) == 3
    assert ops[0].name() == "arith.constant"


def test_function_declaration_call():
    ctx = ir.MLIRContext()
    ctx.load_pyimacs()
    builder = ir.Builder(ctx)


def test_function_extern_call():
    ctx = ir.MLIRContext()
    ctx.load_pyimacs()

    builder = ir.Builder(ctx)
    mod = builder.create_module()
    int_ty = Int.to_ir(builder)

    func_ty = builder.get_llvm_function_ty([int_ty], [int_ty])
    func_decl = builder.get_or_insert_llvm_function(
        mod, "add", func_ty, "public")
    mod.push_back(func_decl)

    func_main = builder.get_or_insert_llvm_function(
        mod, "add_main", func_ty, "public")
    mod.push_back(func_main)
    builder.set_insertion_point_to_start(func_main.add_entry_block())
    arg = func_main.args(0)

    call = builder.llvm_call(func_decl, [arg])
    builder.ret([call.get_result(0)])

    assert mod.has_function("add")
    assert mod.has_function("add_main")


def test_object_type():
    ctx = ir.MLIRContext()
    ctx.load_pyimacs()

    builder = ir.Builder(ctx)
    mod = builder.create_module()
    nil = builder.get_null_as_object()
    print(nil)


def test_function_declaration():
    mod, builder, ctx = init_mod()
    func_name = "get-buffer"
    func_ty = builder.get_llvm_function_ty(
        [builder.get_string_ty()], [builder.get_object_ty()])
    func = builder.get_or_insert_llvm_function(
        mod, func_name, func_ty, "public")
    print(func)
    assert str(func).strip(
    ) == 'llvm.func @"get-buffer"(!lisp.string) -> !lisp.object'
    mod.push_back(func)


test_function_declaration()
