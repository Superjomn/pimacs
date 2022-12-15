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


test_naive_kernel()
