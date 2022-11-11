import ast
from typing import *

import pyimacs._C.libpyimacs.pyimacs as _pyimacs
import pyimacs.lang as pyl


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, function_name: str, gscope: Dict[str, Any]):
        self.function_name = function_name
        self.gscope = gscope
        self.lscope: Dict[str, Any] = dict()
        self.local_defs: Dict[str, pyl.Value] = {}
        self.global_uses: Dict[str, pyl.Value] = {}

    def get_value(self, name: str) -> pyl.Value:
        if name in self.lscope:
            return self.lscope[name]
        if name in self.gscope:
            return self.gscope
        raise ValueError(f'{name} is not defined')

    def set_value(self, name: str, value):
        self.lscope[name] = value
        self.local_defs[name] = value

    def visit_Module(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Return(self, node) -> Any:
        ret_value = self.visit(node.value)
        if ret_value is None:
            self.builder.ret([])
            return None
        if isinstance(ret_value, tuple):
            assert NotImplementedError()
        else:
            ret = pyl.to_value(ret_value, self.builder)
            self.builder.ret([ret.handle])
            return ret.type
