from pyimacs.runtime import jit

from pyimacs import compiler


def test_empty_kernel():
    @jit
    def some_fn(a: int):
        pass

    compiler.compile(some_fn, signature="i -> void")
