import ast
import inspect
import textwrap
from typing import *

import pyimacs.lang as pyl

ir = pyl.ir


class AOTFunction(object):
    def __init__(self, fn, do_not_specialize=None):
        self.fn = fn
        self.module = fn.__module__

        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]
        # specialization hints
        self.do_not_specialize = [] if do_not_specialize is None else do_not_specialize
        self.do_not_specialize = set(
            [self.arg_names.index(arg) if isinstance(arg, str) else arg for arg in self.do_not_specialize])
        # function source code (without decorators
        self.src = textwrap.dedent(inspect.getsource(fn))
        self.src = self.src[self.src.find("def"):]
        # annotations
        print(f"fn.annotation: {fn.__annotations__}")
        self.annotations = {self.arg_names.index(
            name): ty for name, ty in fn.__annotations__.items() if name != "return"}

        self.__annotations__ = fn.__annotations__
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def compile(self) -> str:
        ''' Compile the function to lisp code. '''
        from pyimacs.compiler import compile
        return compile(self.parse())

    def _make_signature(self, sig_key) -> str:
        signature = ",".join([self._type_of(k) for i, k in enumerate(sig_key)])
        return signature

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Cannot call @pyimacs.jit outside of the scope of a kernel")

    def __repr__(self):
        return f"JITFunction<{self.__module__}:{self.__name__}>"


T = TypeVar("T")


def aot(fn: Optional[T] = None,
        do_not_specialize=None):
    def decorator(fn: T) -> AOTFunction:
        assert callable(fn)
        return AOTFunction(fn, do_not_specialize=do_not_specialize)

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
