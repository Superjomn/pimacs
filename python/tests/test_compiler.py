import logging

import pyimacs.lang as pyl
from pyimacs.elisp.buffer import Buffer, buffer_get
from pyimacs.runtime import jit

from pyimacs import compiler


def test_empty_kernel():
    @jit
    def some_fn(a: int):
        pass

    code = compiler.compile(some_fn, signature="i -> void")
    target = '''
(defun some_fn (arg0)
    (let*
        ()
    )
)
    '''
    assert code.strip() == target.strip()


def test_naive_kernel():
    @jit
    def some_fn(a: int):
        b = (a + 1) * 23
        return b + 1

    code = compiler.compile(some_fn, signature="i -> i")
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
    @jit
    def some_fn(a: int):
        a = a + 1
        if True:
            return a
        else:
            # NOTE: Currently, buggy with the return statements within ifOp, we could resolve it by adding a pass to move
            # the statements outside the IfOp(wth return) to else region.
            return a + 1

    code = compiler.compile(some_fn, signature="i -> i")
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

    @jit
    def some_fn():
        return buffer_get("hello")

    code = compiler.compile(some_fn, signature="void -> o")
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
    @jit
    def some_fn():
        buffer = Buffer("*a-buffer*")
        name = buffer.get_name()
        return name

    code = compiler.compile(some_fn, signature="void -> s")
    print(code)

    target = '''
(defun some_fn ()
    (let*
        ()
        (buffer-file-name (buffer-get "*a-buffer*"))
    )
)
    '''
    assert code.strip() == target.strip()
