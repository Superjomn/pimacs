import logging

import pyimacs.lang as pyl
from pyimacs.aot import AOTFunction, aot, get_context
from pyimacs.elisp.buffer import Buffer, _buffer_get
from pyimacs.lang import Int, ir

from pyimacs import compiler

ctx = get_context()


def test_empty_kernel():
    @aot
    def some_fn(a: int):
        pass

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compiler.compile(some_fn, builder=builder, module=module)
    target = '''
(defun some_fn (arg0)
    (let*
        ()
    )
)
    '''
    print(code)
    assert code.strip() == target.strip()


def test_naive_kernel():
    @aot
    def some_fn(a: int) -> int:
        b = (a + 1) * 23
        return b + 1

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compiler.compile(some_fn, builder=builder, module=module)
    print(code)
    target = '''
(defun some_fn (arg0)
    (let*
        ()
        (+ (* (+ arg0 1) 23) 1)
    )
)
    '''
    assert code.strip() == target.strip()


def test_kernel_with_if():
    @aot
    def some_fn(a: int) -> int:
        a = a + 1
        if True:
            return a
        else:
            # NOTE: Currently, buggy with the return statements within ifOp, we could resolve it by adding a pass to move
            # the statements outside the IfOp(wth return) to else region.
            return a + 1

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compiler.compile(some_fn, builder=builder, module=module)
    print(code)

    target = '''
(defun some_fn (arg0)
    (let*
        ()
        (if arg3
            (let*
                ()
                arg2
            )

            (let*
                ()
                (+ (+ arg0 1) 1)
            )
        )
    )
)
    '''
    assert code.strip() == target.strip()


def test_external_call():

    @aot
    def some_fn() -> object:
        return _buffer_get("hello")

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compiler.compile(some_fn, builder=builder, module=module)
    print(code)
    target = '''
(defun some_fn ()
    (let*
        ()
        (buffer-get "hello")
    )
)
    '''
    assert code.strip() == target.strip()


def test_kernel_external_call():
    @aot
    def some_fn(buffer_name: str) -> str:
        buffer = Buffer(buffer_name)
        name = buffer.get_name()
        return name

    builder = ir.Builder(ctx)
    module = builder.create_module()

    code = compiler.compile(some_fn, builder=builder, module=module)
    print(code)

    target = '''
(defun some_fn (arg0)
    (let*
        ()
        (buffer-file-name (buffer-get arg0))
    )
)
    '''
    assert code.strip() == target.strip()


def test_function_with_variaric_argument():
    ctx = ir.MLIRContext()
    ctx.load_pyimacs()
    builder = ir.Builder(ctx)
    mod = builder.create_module()
    func_ty = builder.get_llvm_function_ty(
        [Int.to_ir(builder)], [], is_var_arg=True)
    func = builder.get_or_insert_llvm_function(mod, "add", func_ty, "public")
    mod.push_back(func)

    entry_block = func.add_entry_block()
    builder.set_insertion_point_to_start(entry_block)
    arg0 = func.args(0)
    eq0 = builder.create_add(arg0, builder.get_int32(32))
    builder.ret([eq0])

    builder.set_insertion_point_to_end(entry_block)
    builder.llvm_call(func, [builder.get_int32(32), builder.get_int32(32)])

    print(mod)
