import logging

import pyimacs.lang as pyl
from pyimacs.aot import AOTFunction, aot, get_context
from pyimacs.elisp.buffer import Buffer, buffer_get
from pyimacs.lang import ir

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
        return buffer_get("hello")

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
