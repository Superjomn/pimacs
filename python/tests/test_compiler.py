import pyimacs.lang as pyl
from pyimacs.runtime import jit

from pyimacs import compiler


def test_empty_kernel():
    @jit
    def some_fn(a: int):
        pass

    code = compiler.compile(some_fn, signature="i -> void")
    print(code)


def test_naive_kernel():
    @jit
    def some_fn(a: int):
        b = (a + 1) * 23
        return b + 1

    code = compiler.compile(some_fn, signature="i -> i")
    print(code)


def test_kernel_with_if():
    @jit
    def some_fn(a: int):
        a = a + 1
        if True:
            return a
        # NOTE: Currently, buggy with the return statements within ifOp, we could resolve it by adding a pass to move
        # the statements outside the IfOp(wth return) to else region.
        return a + 1

    code = compiler.compile(some_fn, signature="i -> i")
    print(code)


def test_kernel_external_call():
    @jit
    def some_fn():
        buffer = pyl.Buffer("*a-buffer*")
        name = buffer.name()
        return name

    code = compiler.compile(some_fn, signature="void -> s")
    print(code)


test_kernel_external_call()
