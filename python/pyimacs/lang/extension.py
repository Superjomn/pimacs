__all__ = [
    'register_extern',
    'builder',
    'ctx',
    'module',
    'Ext',
]

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import *

from pyimacs.lang.core import DataType, Object
from pyimacs.lang.semantic import ir, pl

global_mlir_module = None
global_mlir_builder = None
global_mlir_ctx = None
global_extern_functions: Dict[str, ir.Function] = {}


# Usage:
# @register_extern("buffer-get")
# def buffer_get(buf_name: str) -> object: ...

def set_global_builder(builder: ir.Builder):
    global global_mlir_builder
    global_mlir_builder = builder


def set_global_module(module: ir.Module):
    global global_mlir_module
    global_mlir_module = module


def builder() -> ir.Builder:
    assert global_mlir_builder, "Builder is not set."
    return global_mlir_builder


def module() -> ir.Module:
    assert global_mlir_module, "Module is not set."
    return global_mlir_module


def ctx() -> ir.MLIRContext:
    global global_mlir_ctx
    if not global_mlir_ctx:
        global_mlir_ctx = ir.MLIRContext()
        global_mlir_ctx.load_pyimacs()
    return global_mlir_ctx


def register_extern(func_name: str):
    ''' Register an external function into the module. This would help in MLIR call. '''
    return functools.partial(_register_extern, func_name=func_name)


@dataclass
class Signature:
    ''' Record signature and some other meta information for an external function. '''
    elisp_func_name: str  # the corresponding elisp function name
    signature: inspect.FullArgSpec
    file_name: str
    lineno: str

    @staticmethod
    def get_func_id(signature: "Signature"):
        ''' Generate a unique id for the function. '''
        return f"{signature.elisp_func_name}_{signature.file_name}:{signature.lineno}"


# Holds all the signatures for the extern functions.
# A map from func_id to Signature.
func_signature_registry: Dict[str, Signature] = dict()


def _register_extern(func: Callable, func_name: str):
    signature = Signature(func_name, inspect.getfullargspec(
        func), inspect.getfile(func), inspect.getsourcelines(func)[1])
    func_id = Signature.get_func_id(signature)

    func_signature_registry[func_id] = signature

    def register_to_module(signature: inspect.FullArgSpec, module: ir.Module, builder: ir.Builder):
        annotation = signature.annotations
        if not module.has_function(func_name):
            ''' Register function into module. '''
            # TODO[Superjomn]: Consider tuple later
            ret_type = [dtype_to_mlir_type(
                annotation['return'], builder)] if annotation['return'] is not None else []
            # get function type
            in_types = []
            for arg, py_ty in annotation.items():
                if arg == "return":
                    continue
                in_types.append(dtype_to_mlir_type(py_ty, builder))
            print('in_types', in_types)
            is_var_arg = bool(signature.varargs)
            func_ty = builder.get_llvm_function_ty(
                in_types, ret_type, is_var_arg)

            func = builder.get_or_insert_llvm_function(
                module, func_name, func_ty, "public")

            module.push_back(func)

    def fn(*args, **kwargs):
        assert module(), "module() need to be set before calling extern function."
        assert builder(), "builder() need to be set before calling extern function."
        assert not kwargs, "kwargs is not supported in extern function yet."

        func_id = fn.__doc__
        assert func_id

        signature = func_signature_registry.get(func_id).signature
        assert signature
        annotation = signature.annotations
        decl_args = signature.args

        if not module().has_function(func_name):
            try:
                register_to_module(signature, module(), builder())
            except TypeError:
                logging.error(f"Failed to register function: {func_name} with signature: {signature}")
                raise

        mlir_args = []
        for idx, arg in enumerate(args):
            py_type = annotation.get(decl_args[idx])
            mlir_arg = arg_to_mlir(arg, py_type)
            mlir_args.append(mlir_arg)

        func = module().get_llvm_function(func_name)
        called = builder().llvm_call(func, mlir_args)
        if called.get_num_results() == 0:
            return
        assert called.get_num_results() == 1, "Only one result is supported now."
        return called.get_result(0)

    # We put the func_id into the docstring of the function to make it possible to obtain the signature later.
    fn.__doc__ = Signature.get_func_id(signature)

    return fn


def dtype_to_mlir_type(dtype: Any, builder: ir.Builder) -> ir.Type:
    # TODO[Superjomn]: Turn the following ifs to a map.
    if dtype is int:
        return builder.get_int32_ty()
    if dtype is float:
        return builder.get_float_ty()
    if dtype is str:
        return builder.get_string_ty()
    if dtype is object:
        return builder.get_object_ty()
    if dtype is bool:
        return builder.get_int1_ty()

    raise NotImplementedError()


class Ext:
    '''
    Placeholder to mark the externsion classes. All the inherient class's method will be evaluated in compile time.
    '''

    def __handle_return__(self): ...
    ''' handler to processing return in compiler. '''

    def __handle_assign_subscript__(self, key, value):
        ''' handler to processing: x[key] = v. '''
        raise NotImplementedError()


def _arg_to_mlir_value(v: Any):
    if type(v) is pl.Value:
        v = v.handle
    if type(v) is ir.Value:
        return v
    if type(v) is ir.Operation:
        return v.to_value()

    if type(v) is str:
        return builder().get_string(v)
    if type(v) is int:
        return builder().get_int32(v)
    if type(v) is float:
        return builder().get_float32(v)
    if type(v) is object:
        return v
    if type(v) is list or type(v) is tuple:
        return [_arg_to_mlir_value(e) for e in v]

    raise NotImplementedError(f"{v} of type {type(v)} is not supported")


def arg_to_mlir(arg, type=None):
    if arg is None:
        assert type is not None, "type is required when arg is None."
        if type is str:
            return builder().get_null_as_string()
        if type is int:
            return builder().get_null_as_int()
        if type is float:
            return builder().get_null_as_float()
        if type is object:
            return builder().get_null_as_object()
        if type is bool:
            return builder().get_null_as_bool()
        raise NotImplementedError(f"{arg} of {type}")
    else:
        return _arg_to_mlir_value(arg)
