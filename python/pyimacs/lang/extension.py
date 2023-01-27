import functools
import inspect
from dataclasses import dataclass
from typing import *

from pyimacs.lang.core import DataType, Object
from pyimacs.lang.semantic import *

global_mlir_module = None
global_mlir_builder = None
global_mlir_ctx = None
global_extern_functions: Dict[str, ir.Function] = {}


def builder() -> ir.Builder:
    global global_mlir_builder
    if not global_mlir_builder:
        global_mlir_builder = ir.Builder(ctx())
    return global_mlir_builder


def module() -> ir.Module:
    global global_mlir_module
    if not global_mlir_module:
        global_mlir_module = builder().create_module()
    return global_mlir_module


def ctx() -> ir.MLIRContext:
    global global_mlir_ctx
    if not global_mlir_ctx:
        global_mlir_ctx = ir.MLIRContext()
        global_mlir_ctx.load_pyimacs()
    return global_mlir_ctx


def register_extern(func_name: str):
    return functools.partial(_register_extern, func_name=func_name)


def _register_extern(func: Callable, func_name: str):
    signature = inspect.getfullargspec(func)
    args = signature.args
    if not module().has_function(func_name):
        annotation = signature.annotations
        ''' Register function into module. '''
        # TODO[Superjomn]: Consider tuple later
        ret_type = dtype_to_mlir_type(annotation['return'], builder())
        # get function type
        in_types = []
        for arg, py_ty in annotation.items():
            if arg == "return":
                continue
            in_types.append(dtype_to_mlir_type(py_ty, builder()))
        func_ty = builder().get_function_ty(in_types, [ret_type])
        builder().get_or_insert_function(module(), func_name, func_ty, "public")

    def arg_to_mlir_value(v: Any):
        if type(v) is str:
            return builder().get_string(v)
        if type(v) is int:
            return builder().get_int32(v)
        if type(v) is float:
            return builder().get_float32(v)
        if type(v) is object:
            return v
        raise NotImplementedError()

    def fn(*args, **kwargs):
        assert not kwargs, "kwargs is supported yet"
        args = [arg_to_mlir_value(arg) for arg in args]
        return builder().call(module().get_function(func_name), args)

    return fn


def dtype_to_mlir_type(dtype: Any, builder: ir.Builder) -> ir.Type:
    if dtype is int:
        return builder.get_int32_ty()
    if dtype is float:
        return builder.get_float_ty()
    if dtype is str:
        return builder.get_string_ty()
    if dtype is object:
        return builder.get_object_ty()
    assert NotImplementedError()



@register_extern("buffer-get")
def buffer_get(buf_name: str) -> object: ...


print(buffer_get)

print(buffer_get("hello"))

# def _register_extern_function(func_name: str, func_ty: ir.Type, builder: ir.Builder, mod: ir.Module):
# global_extern_functions[func_name] =
